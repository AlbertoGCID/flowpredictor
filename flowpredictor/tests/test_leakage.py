"""
Regression tests guarding against temporal data leakage: during inference at step t,
neither the seq2seq decoder nor the Random Forest meta-learner may access the observed
values of Qe(t+1) or Qe(t+2) -- the very quantities being forecast.

Covers the matrix-construction / inference code paths in
resultados/prueba63/src/pipeline/iteration.py:

  1. preprocessing.sliding_window / prepare_model_inputs, used by
     Iteration._build_windows_from_dfs to build encoder/decoder windows.
  2. Iteration._predict_iterative, the autoregressive decoding loop used at
     inference time.
  3. Iteration.train_classifier / predict_classifier, which feed encoder+decoder
     windows (plus base-model predictions) into the Random Forest meta-learner.

evaluator.py is not exercised here: it only reads back cached prediction .npz files
already produced by Iteration and performs no inference or matrix construction of its
own, so it has no code path through which Qe(t+1)/Qe(t+2) could leak into a model.
"""
from __future__ import annotations

import os
import types

import numpy as np
import pandas as pd
import pytest

from resultados.prueba63.src.pipeline.preprocessing import sliding_window, prepare_model_inputs


# --------------------------------------------------------------------------- #
# 1. Matrix construction: sliding_window / prepare_model_inputs
# --------------------------------------------------------------------------- #

def _make_synthetic_df(n_days: int, band_offset: float = 0.0) -> pd.DataFrame:
    """Deterministic frame where every column lives in its own, non-overlapping
    numeric band, so any leaked future Qe value is unambiguously identifiable.
    band_offset shifts every band uniformly, so two calls with different offsets
    produce mutually disjoint value ranges (used to keep a synthetic "test" split
    numerically distinguishable from "train")."""
    t = np.arange(n_days, dtype=float) + band_offset
    return pd.DataFrame({
        "Fecha": pd.date_range("2020-01-01", periods=n_days, freq="D"),
        "Qe": 1000.0 + t,            # historico + target column
        "rain1": -1.0 * t,           # unrelated historico
        "pred_l/m2": 5000.0 + t,     # 24h rain forecast
        "pred_l/m2_2d": 6000.0 + t,  # 48h rain forecast
        "pred_l/m2_3d": 7000.0 + t,  # 72h rain forecast
    })


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_sliding_window_decoder_never_contains_future_qe(offset):
    """Decoder rows built by sliding_window() must only carry the *last known*
    Qe (i.e. Qe(t)) in their first column -- never the observed Qe(t+1)/Qe(t+2)."""
    df = _make_synthetic_df(n_days=60)
    contextos = 10

    x_enc, x_dec, y = sliding_window(
        historicos=df[["Qe", "rain1"]],
        predicciones=df[["pred_l/m2", "pred_l/m2_2d", "pred_l/m2_3d"]],
        labels=df["Qe"],
        output_column="Qe",
        input_width=contextos,
        offset=offset,
    )
    assert len(x_enc) > 0
    qe_col = 0  # "Qe" is first in historicos

    for i in range(len(x_enc)):
        ultimo_qe = df["Qe"].iloc[i + contextos - 1]      # Qe(t), known at emission day
        forbidden = {df["Qe"].iloc[i + contextos]}         # Qe(t+1)
        if i + contextos + 1 < len(df):
            forbidden.add(df["Qe"].iloc[i + contextos + 1])  # Qe(t+2)

        # Encoder window spans only [i, i+contextos): must not contain future Qe.
        assert not (forbidden & set(x_enc[i][:, qe_col].tolist()))

        # Decoder's Qe channel repeats only the known last observed value.
        decoder_qe_values = set(x_dec[i][:, 0].tolist())
        assert decoder_qe_values == {ultimo_qe}
        assert not (forbidden & decoder_qe_values)

        # The realized target itself must never appear in encoder/decoder input.
        assert y[i][0] not in x_enc[i][:, qe_col]
        assert y[i][0] not in x_dec[i][:, 0]


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_prepare_model_inputs_decoder_never_contains_future_qe(offset):
    """Same guarantee, exercised through prepare_model_inputs(), the exact
    function Iteration._build_windows_from_dfs() calls to build train/test windows."""
    df = _make_synthetic_df(n_days=60)
    contextos = 10

    (x_tr_enc, x_tr_dec, y_tr, _), _, _ = prepare_model_inputs(
        train_norm=df, val_norm=None, test_norm=df,
        variable_salida="Qe",
        historicos=["Qe", "rain1"],
        predicciones=["pred_l/m2", "pred_l/m2_2d", "pred_l/m2_3d"],
        contextos=contextos, offset=offset,
    )
    assert x_tr_enc.shape[0] > 0
    qe_idx = 0

    for i in range(x_tr_enc.shape[0]):
        ultimo_qe = x_tr_enc[i, -1, qe_idx]
        # Every decoder step repeats only the last known Qe, matching the encoder tail.
        assert np.all(x_tr_dec[i, :, 0] == ultimo_qe)
        # The realized target is never present anywhere in encoder or decoder input.
        target = y_tr[i, 0]
        assert target not in x_tr_enc[i, :, qe_idx]
        assert target not in x_tr_dec[i, :, 0]
        assert target > ultimo_qe  # sanity: synthetic Qe series is monotonic increasing


def test_sliding_window_strict_causality():
    """Adversarial test of the REAL sliding_window() (no mocks) for offset=1, the production case.

    y is the FULL sequence [Qe(t+1), Qe(t+2)] and the decoder carries 2 steps
    with the real rainfall forecast of EACH day. Exact invariants are checked,
    not only "the forbidden value is absent":
      1. y[i] == [Qe(t+1), Qe(t+2)] exactly, in that order.
      2. The decoder has 2 steps: step 0 with pred_l/m2 (24h, day t+1) and
         step 1 (last) with pred_l/m2_2d (48h, day t+2) -- neither pred_l/m2
         nor pred_l/m2_3d in the final step.
    """
    n_days = 100
    t = np.arange(n_days, dtype=float)
    df = pd.DataFrame({
        "Fecha": pd.date_range("2020-01-01", periods=n_days, freq="D"),
        "Qe": np.sin(t) + t,
        "rain1": -1.0 * t,
        "pred_l/m2": np.cos(t),
        "pred_l/m2_2d": np.cos(t) + 1000.0,
        "pred_l/m2_3d": np.cos(t) + 2000.0,
    })
    contextos, offset = 10, 1

    x_enc, x_dec, y = sliding_window(
        historicos=df[["Qe", "rain1"]],
        predicciones=df[["pred_l/m2", "pred_l/m2_2d", "pred_l/m2_3d"]],
        labels=df["Qe"],
        output_column="Qe",
        input_width=contextos,
        offset=offset,
    )
    assert len(y) > 0

    for i in range(len(y)):
        ultimo_dia_idx = i + contextos - 1   # last encoder day == issue day (t)
        target_idx = i + contextos + offset  # index of the LAST target step in df

        # 1. y[i] == [Qe(t+1), Qe(t+2)] exactly.
        assert target_idx == ultimo_dia_idx + 2
        assert y[i].shape == (offset + 1,)
        assert y[i][0] == pytest.approx(df["Qe"].iloc[ultimo_dia_idx + 1])   # Qe(t+1)
        assert y[i][1] == pytest.approx(df["Qe"].iloc[target_idx])           # Qe(t+2)

        # 2. offset=1 -> 2 decoder steps: step0=24h, step1 (last)=48h.
        assert x_dec[i].shape[0] == offset + 1
        rain_step0 = x_dec[i][0, 1]
        rain_step1 = x_dec[i][1, 1]
        val_24h = df["pred_l/m2"].iloc[ultimo_dia_idx]
        val_48h = df["pred_l/m2_2d"].iloc[ultimo_dia_idx]
        val_72h = df["pred_l/m2_3d"].iloc[ultimo_dia_idx]
        assert rain_step0 == pytest.approx(val_24h)
        assert rain_step1 == pytest.approx(val_48h)
        assert rain_step1 != pytest.approx(val_24h)
        assert rain_step1 != pytest.approx(val_72h)


def test_rf_strict_functional_independence():
    """The encoder/decoder features X produced by sliding_window() must be a pure
    function of historicos/predicciones, functionally INDEPENDENT of any
    transformation applied to the labels. If X changed when y is transformed,
    sliding_window() would leak target information into the features -- the
    kind of leak the downstream RF would silently inherit. The real
    sliding_window() is called twice with different labels and fixed inputs,
    and X is compared byte by byte."""
    n = 60
    t = np.arange(n, dtype=float)
    historicos = pd.DataFrame({"Qe": t, "rain1": -t})
    predicciones = pd.DataFrame({
        "pred_l/m2": t + 100.0,
        "pred_l/m2_2d": t + 200.0,
        "pred_l/m2_3d": t + 300.0,
    })
    y_true = pd.Series(t, name="Qe")

    common_kwargs = dict(
        historicos=historicos, predicciones=predicciones,
        output_column="Qe", input_width=10, offset=1,
    )
    x_enc_base, x_dec_base, y_base = sliding_window(labels=y_true, **common_kwargs)

    # Arbitrary non-linear label transformation (t/20 <= ~3 avoids exp overflow).
    y_true_fake = np.exp(y_true / 20.0)
    x_enc_fake, x_dec_fake, y_fake = sliding_window(labels=y_true_fake, **common_kwargs)

    # The features must NOT change a single value when y is transformed.
    np.testing.assert_array_equal(x_enc_base, x_enc_fake)
    np.testing.assert_array_equal(x_dec_base, x_dec_fake)

    # Negative control: y MUST reflect the transformation (otherwise both calls
    # would be identical and the test trivial).
    assert not np.allclose(y_base, y_fake)


# --------------------------------------------------------------------------- #
# 2. Iterative decoding at inference time: Iteration._predict_iterative
# --------------------------------------------------------------------------- #

def test_predict_iterative_never_reads_future_observations():
    """Autoregressive decoding must only ever consume (a) the precomputed decoder
    window (leak-free per the tests above) or (b) the model's own previous
    prediction -- never an externally observed 'true future' array. A refactor
    that threads real y_true into this loop (e.g. accidental teacher forcing)
    would fail this test."""
    import tensorflow as tf
    from resultados.prueba63.src.pipeline.iteration import Iteration

    batch = 2
    KNOWN_QE = 999.0      # Qe(t): the single value the decoder is allowed to see
    FORBIDDEN_QE = 555.0  # stand-in for a real Qe(t+1)/Qe(t+2); must never appear

    # offset=2 -> sliding_window() produces (offset+1)=3 decoder steps with known
    # data (one per target day [t+1, t+2, t+3]). overstep=2 is set to a NON-ZERO
    # value on purpose, to check that it is ignored: _predict_iterative() never
    # generates steps beyond offsets+1, so total_steps == 3 regardless of
    # self.p['overstep'].
    dummy_self = types.SimpleNamespace(p={"offsets": 2, "overstep": 2})

    x_enc = np.zeros((batch, 3, 2), dtype=np.float32)
    x_dec = np.array([
        [[KNOWN_QE, 10.0, 11.0, 12.0, 13.0],
         [KNOWN_QE, 20.0, 21.0, 22.0, 23.0],
         [KNOWN_QE, 30.0, 31.0, 32.0, 33.0]],
        [[KNOWN_QE, 10.0, 11.0, 12.0, 13.0],
         [KNOWN_QE, 20.0, 21.0, 22.0, 23.0],
         [KNOWN_QE, 30.0, 31.0, 32.0, 33.0]],
    ], dtype=np.float32)

    captured_inputs = []
    call_counter = {"n": 0}

    def encoder_model(x, training=False):
        return tf.zeros((batch, 4)), tf.zeros((batch, 4))

    def decoder_model(inputs, training=False):
        current_input, state_h, state_c = inputs
        captured_inputs.append(current_input.numpy().copy())
        step = call_counter["n"]
        call_counter["n"] += 1
        pred = tf.fill([batch, 1, 1], float(1000 + step))  # distinguishable own prediction
        return pred, state_h, state_c

    preds = Iteration._predict_iterative(dummy_self, encoder_model, decoder_model, x_enc, x_dec)

    # Exactly offsets+1=3 predictions, NEVER offsets+1+overstep=5.
    assert preds.shape == (batch, 3, 1)
    assert len(captured_inputs) == 3

    # step=0 starts from the real precomputed decoder row (leak-free).
    assert np.all(captured_inputs[0][:, :, 0] == KNOWN_QE)
    # step>=1: unconditional autoregressive feedback -- the Qe channel of the
    # input is ALWAYS the model's own previous prediction, never the real future
    # value nor the constant KNOWN_QE. The decoder stub returns 1000+step
    # (0-based call index), so the input of step k>=1 carries exactly
    # 1000+(k-1) in the Qe channel.
    assert np.all(captured_inputs[1][:, :, 0] == 1000.0)
    assert np.all(captured_inputs[2][:, :, 0] == 1001.0)

    # Across the whole loop the forbidden sentinel never appears anywhere.
    for inp in captured_inputs:
        assert not np.any(inp == FORBIDDEN_QE)


# --------------------------------------------------------------------------- #
# 3. Random Forest meta-learner: Iteration.train_classifier / predict_classifier
# --------------------------------------------------------------------------- #

def test_random_forest_inputs_exclude_observed_future_qe(tmp_path, monkeypatch):
    """clf.fit()/clf.predict() must be fed only encoder/decoder windows -- built
    here via the REAL sliding_window()/prepare_model_inputs() pipeline (no mocks
    bypassing them) -- and base-model *predictions*; the observed
    ground truth train_classifier()/predict_classifier() use to derive
    target_indices (the npz 'y_true' field) must never appear inside the
    flattened feature matrix at fit or predict time."""
    from resultados.prueba63.src.pipeline.iteration import Iteration
    import resultados.prueba63.src.models.randomforest as rf_mod

    params = {
        "algorithms": "rf_regressor",
        "loss_name": "original_mae",
        "numero_prueba": "leaktest",
        "result_path": str(tmp_path),
        "offsets": 1,
        "contextos": 5,
        "steps": 1,
        "seed": 0,
        # CONFIG's defaults for these are lists (grid-search values); Iteration's
        # _scalar() only unwraps lists passed explicitly, not CONFIG's own
        # fallback default, so they must be given here as plain scalars.
        "lr": 0.001,
        "ano_test": 2017,
        "batch_size": 8,
        "max_epochs": 1,
        "max_epochs_clasificador": 1,
        "overstep": 0,
        "coef_de_pond": 0.5,
        "umbrales": 0,
        "dropout": False,
        "l2_options": False,
    }
    it = Iteration(params)
    # predict_classifier() requires the train-fixed p90 threshold to exist
    # (anti-leakage: no dynamic np.quantile(y_true) fallback on test);
    # this test doesn't call ensure_and_load_data(), so it must be seeded directly.
    it._norm_params = {"Qe": {"p90": 0.0}}

    # --- Real, leak-free encoder/decoder windows via prepare_model_inputs() --
    # the exact function Iteration._build_windows_from_dfs() calls in production.
    # df_test uses a disjoint numeric band so its Qe/rain values can never
    # collide with df_train's.
    contextos = params["contextos"]
    df_train = _make_synthetic_df(n_days=40)
    df_test = _make_synthetic_df(n_days=25, band_offset=10_000.0)
    window_kwargs = dict(
        variable_salida="Qe", historicos=["Qe", "rain1"],
        predicciones=["pred_l/m2", "pred_l/m2_2d", "pred_l/m2_3d"],
        contextos=contextos, offset=params["offsets"],
    )
    (x_tr_enc, x_tr_dec, y_tr, tr_dates), _, _ = prepare_model_inputs(
        train_norm=df_train, val_norm=None, test_norm=df_train, **window_kwargs
    )
    _, _, (x_te_enc, x_te_dec, y_te, te_dates) = prepare_model_inputs(
        train_norm=df_train, val_norm=None, test_norm=df_test, **window_kwargs
    )
    N_TR, N_TE = x_tr_enc.shape[0], x_te_enc.shape[0]
    assert N_TR >= 8 and N_TE >= 4  # enough samples to cover several OOF windows

    it.train_inputs = (x_tr_enc, x_tr_dec, y_tr, tr_dates)
    it.test_inputs = (x_te_enc, x_te_dec, y_te, te_dates)

    def fake_build_windows_normal():
        it._windows_tag = "normal"
        return it.train_inputs, it.test_inputs

    it.build_windows_normal = fake_build_windows_normal

    # --- Fake base-model predictions: "observed" values live in a forbidden band ---
    bases = ["original_mae_0", "pinball_weighted_from_penalty_2"]
    y_true_train = 10000.0 + np.arange(N_TR)   # stand-in for observed Qe(t+1)/Qe(t+2)
    y_true_test = 20000.0 + np.arange(N_TE)
    train_preds = {bases[0]: 200.0 + np.arange(N_TR), bases[1]: 210.0 + np.arange(N_TR)}
    test_preds = {bases[0]: 300.0 + np.arange(N_TE), bases[1]: 310.0 + np.arange(N_TE)}

    def _write_pred_npz(path, y_true, y_pred, dates):
        np.savez(path, y_true=y_true, y_pred=y_pred,
                 dates=pd.to_datetime(pd.Index(list(dates))).values)

    # train_classifier() checks that the dates of self.train_inputs (tr_dates)
    # and those stored in the prediction .npz describe the SAME windows
    # (anti-shuffling guard), so the fake must use the real dates from
    # prepare_model_inputs(), not an arbitrary pd.date_range.
    pred_dir = it.paths.predictions_cache
    train_npz_paths = {}
    test_npz_paths = {}
    for b in bases:
        p_tr = os.path.join(pred_dir, f"pred_{b}_train.npz")
        _write_pred_npz(p_tr, y_true_train, train_preds[b], tr_dates)
        train_npz_paths[b] = p_tr

        p_te = os.path.join(pred_dir, f"pred_{b}.npz")
        _write_pred_npz(p_te, y_true_test, test_preds[b], te_dates)
        test_npz_paths[b] = p_te

    def fake_ensure_predictions(force=False, save_csv=True, subset="test"):
        paths = train_npz_paths if subset == "train" else test_npz_paths
        return [{"base": b, "npz": paths[b]} for b in bases]

    it.ensure_predictions = fake_ensure_predictions

    # Nested blocked out-of-fold: train_classifier() calls
    # ensure_oof_train_predictions() (not ensure_predictions(subset="train"));
    # it is faked directly to avoid depending on self._train_df / self.p["cv_strategy"].
    it.ensure_oof_train_predictions = lambda force=False: [
        {"base": b, "npz": train_npz_paths[b]} for b in bases
    ]

    # train_classifier() reads target_indices directly from this npz (the
    # OFFICIAL label of ensure_ensemble_labels()), so the fake must include
    # 'indices' with the same length as base_preds_train (N_TR), like the real npz.
    ensemble_indices = (np.arange(N_TR) % len(bases)).astype(np.int32)
    ensemble_npz = os.path.join(it.paths.ensemble, "fake_ensemble_train.npz")
    np.savez(ensemble_npz, models=np.array(bases, dtype=object), indices=ensemble_indices)
    it.ensure_ensemble_labels = lambda subset="test", force=False: {"npz": ensemble_npz}

    # --- Spy on the real RF wrapper's fit/predict to capture exactly what it receives ---
    captured = {"fit": [], "predict": []}
    orig_fit = rf_mod.RFRegressorWrapper.fit
    orig_predict = rf_mod.RFRegressorWrapper.predict

    def spy_fit(self, inputs, y, **kwargs):
        captured["fit"].append(inputs)
        return orig_fit(self, inputs, y, **kwargs)

    def spy_predict(self, inputs):
        captured["predict"].append(inputs)
        return orig_predict(self, inputs)

    monkeypatch.setattr(rf_mod.RFRegressorWrapper, "fit", spy_fit)
    monkeypatch.setattr(rf_mod.RFRegressorWrapper, "predict", spy_predict)

    it.train_classifier(force=False)
    it.predict_classifier(force=False)

    assert len(captured["fit"]) == 1
    assert len(captured["predict"]) == 1

    def _flatten(inputs):
        return np.concatenate([np.asarray(a).reshape(np.asarray(a).shape[0], -1) for a in inputs], axis=1)

    fit_matrix = _flatten(captured["fit"][0])
    predict_matrix = _flatten(captured["predict"][0])

    # clf.fit() sees every OOF sample; the fake ensure_oof_train_predictions
    # covers all N_TR dates of tr_dates, so fit sees all N_TR rows.
    assert fit_matrix.shape[0] == N_TR

    forbidden = set(y_true_train.tolist()) | set(y_true_test.tolist())
    assert not (forbidden & set(fit_matrix.flatten().tolist()))
    assert not (forbidden & set(predict_matrix.flatten().tolist()))


# --------------------------------------------------------------------------- #
# 4. Multi-step Seq2Seq end to end (sliding_window -> train_step ->
#    _predict_iterative), offset=0/1/2.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("offset", [0, 1, 2])
def test_seq2seq_multistep_end_to_end_sliding_window_to_predict_iterative(offset):
    """Exercise the three real multi-step components (no mocks).

      1. sliding_window(offset=k) produces x_dec/y with exactly (k+1) steps.
      2. train_step() backpropagates over the full sequence without failures
         or non-finite gradients -- a real backprop test, not a shape-only
         smoke test.
      3. Iteration._predict_iterative() produces (offset+1) predictions from the
         SAME x_dec, usable with step_idx=offset (the final horizon).
    """
    import tensorflow as tf
    from resultados.prueba63.src.models.models import build_encoder, build_decoder
    from resultados.prueba63.src.models.training import train_step
    from resultados.prueba63.src.pipeline.iteration import Iteration

    # Normalized range (~[0, 1], like the real data after normalize_with_params),
    # NOT large raw values: the test was designed when the decoder output layer
    # used ReLU, where large-scale inputs push the pre-activation into the dead
    # zone for any seed. The output layer is now linear; the realistic scale is kept.
    n_days = 60
    t = np.linspace(0.0, 1.0, n_days)
    df = pd.DataFrame({
        "Qe": 0.3 + 0.2 * np.sin(2 * np.pi * t),
        "rain1": 0.5 - 0.3 * t,
        "pred_l/m2": 0.4 + 0.1 * np.cos(2 * np.pi * t),
        "pred_l/m2_2d": 0.45 + 0.1 * np.cos(2 * np.pi * t),
        "pred_l/m2_3d": 0.5 + 0.1 * np.cos(2 * np.pi * t),
    })
    contextos = 10
    n_steps = offset + 1

    # --- 1. real sliding_window: correct dimensions ---
    x_enc, x_dec, y = sliding_window(
        historicos=df[["Qe", "rain1"]],
        predicciones=df[["pred_l/m2", "pred_l/m2_2d", "pred_l/m2_3d"]],
        labels=df["Qe"], output_column="Qe", input_width=contextos, offset=offset,
    )
    assert x_enc.ndim == 3 and x_enc.shape[1] == contextos
    assert x_dec.ndim == 3 and x_dec.shape[1] == n_steps and x_dec.shape[2] == 5
    assert y.ndim == 2 and y.shape[1] == n_steps
    assert x_enc.shape[0] == x_dec.shape[0] == y.shape[0] > 0

    # seed=1 / encoder_units=decoder_units=16: empirically verified combination
    # for the three parameterized offsets.
    it_p = {"seed": 1, "dropout": False, "l2_options": False,
            "encoder_units": 16, "decoder_units": 16}
    n_enc_feat, n_dec_feat = x_enc.shape[2], x_dec.shape[2]
    encoder_model, _, _, _ = build_encoder((contextos, n_enc_feat), it_p)
    decoder_model = build_decoder(n_dec_feat, it_p)

    x_enc_tf = tf.constant(x_enc, dtype=tf.float32)
    x_dec_tf = tf.constant(x_dec, dtype=tf.float32)
    y_tf = tf.constant(y, dtype=tf.float32)

    # --- 2. real train_step: backprop over the full sequence ---
    iteration_params = {"offsets": offset, "overstep": 0, "loss_name": "original_mae",
                         "penalty": 0.0, "k_high": None, "tau": None}
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    weights_before = [w.numpy().copy() for w in
                       encoder_model.trainable_variables + decoder_model.trainable_variables]

    loss_value = train_step(encoder_model, decoder_model, x_enc_tf, x_dec_tf, y_tf,
                             optimizer, iteration_params)

    assert np.isfinite(loss_value.numpy()), "The loss is not finite -- backprop is broken"
    weights_after = [w.numpy() for w in
                      encoder_model.trainable_variables + decoder_model.trainable_variables]
    assert any(not np.allclose(a, b) for a, b in zip(weights_before, weights_after)), (
        "No weight changed after train_step() -- the gradient did not reach the parameters "
        "(backpropagation may be cut in the multi-step unrolling)."
    )

    # --- 3. real _predict_iterative: (offset+1) predictions from the same x_dec ---
    dummy_self = types.SimpleNamespace(p={"offsets": offset, "overstep": 0})
    preds_seq = Iteration._predict_iterative(dummy_self, encoder_model, decoder_model,
                                              x_enc, x_dec)
    assert preds_seq.shape == (x_enc.shape[0], n_steps, 1)
    assert np.all(np.isfinite(preds_seq))

    # The final evaluated horizon (step_idx=offset, see Iteration.ensure_predictions)
    # must be a valid index into the generated sequence.
    step_idx = offset
    assert 0 <= step_idx < preds_seq.shape[1]


# --------------------------------------------------------------------------- #
# 5. Adversarial graph test: compute_loss_by_name() must never return a
#    (batch, steps, steps) matrix through cross broadcasting.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("loss_name,kwargs", [
    ("original_mae", {}),
    ("custom_loss_mae", {"penalty_value": 0.5}),
    ("pinball_from_penalty", {"penalty_value": 4.0}),
    ("pinball_weighted_from_penalty", {"penalty_value": 4.0, "k_high": 2.0}),
])
def test_compute_loss_by_name_returns_scalar_never_cross_broadcast_matrix(loss_name, kwargs):
    """With correctly shaped y and predictions (batch, steps, 1),
    compute_loss_by_name() must ALWAYS return a scalar (reduce_mean over the
    whole batch/sequence), never a vector and above all never a
    (batch, steps, steps) matrix from cross broadcasting between a wrongly
    expanded y ((batch, 1, steps) instead of (batch, steps, 1)) and predictions."""
    import tensorflow as tf
    from resultados.prueba63.src.models.training import compute_loss_by_name

    batch, steps = 4, 3
    y = tf.random.uniform((batch, steps, 1), seed=0)
    predictions = tf.random.uniform((batch, steps, 1), seed=1)

    loss = compute_loss_by_name(loss_name, y, predictions, **kwargs)

    assert loss.shape.ndims == 0, (
        f"{loss_name}: expected a scalar (rank 0), got shape {loss.shape} -- "
        f"possible (batch,steps,steps) matrix from cross broadcasting."
    )
    assert np.isfinite(float(loss.numpy()))


def test_train_step_shape_guard_rejects_mismatched_sequence_length():
    """If x_dec and y arrive misaligned (different number of steps, e.g. an
    upstream bug that truncates or repeats rows), train_step() must fail LOUDLY
    through tf.debugging.assert_shapes instead of letting TF silently
    cross-broadcast and train on a corrupted loss. Direct proof that the shape
    guard is a real block, not decorative."""
    import tensorflow as tf
    from resultados.prueba63.src.models.models import build_encoder, build_decoder
    from resultados.prueba63.src.models.training import train_step

    contextos, n_enc_feat, n_dec_feat, batch = 6, 3, 5, 4
    it_p = {"seed": 1, "dropout": False, "l2_options": False, "encoder_units": 8, "decoder_units": 8}
    encoder_model, _, _, _ = build_encoder((contextos, n_enc_feat), it_p)
    decoder_model = build_decoder(n_dec_feat, it_p)

    offset = 1  # -> train_step unrolls 2 steps (offsets+1)
    x_enc = tf.random.uniform((batch, contextos, n_enc_feat))
    x_dec = tf.random.uniform((batch, offset + 1, n_dec_feat))  # 2 steps, correct
    y_mismatched = tf.random.uniform((batch, offset + 2))       # 3 steps, MISALIGNED on purpose

    iteration_params = {"offsets": offset, "overstep": 0, "loss_name": "original_mae",
                         "penalty": 0.0, "k_high": None, "tau": None}
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    with pytest.raises(Exception):
        train_step(encoder_model, decoder_model, x_enc, x_dec, y_mismatched,
                   optimizer, iteration_params)
