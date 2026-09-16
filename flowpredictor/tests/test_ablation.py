"""Tests of the five ablation configurations (see ABLATION_CONFIGS in main_pipeline.py).

  M1: no pretraining, symmetric MAE (penalty=[0]), no RF
  M2: no pretraining, asymmetric pinball (penalty=[2]), no RF
  M3: pretraining,    symmetric MAE (penalty=[0]), no RF
  M4: pretraining,    asymmetric pinball (penalty=[2]), no RF
  M5: pretraining,    asymmetric pinball (penalty=[0,2,4,6,8,10]), RF

M2/M4 use a single penalty (not the full M5 grid): with use_rf=False,
_run_single_config only evaluates the first valid grid combination
(penalty=2), so training 4/6/8/10 would be computation without a consumer.

Only the finetune tests train real (synthetic, 1-epoch) models; the rest check
that build_ablation_configs() yields exactly these five parameter sets and that
Iteration() + build_encoder/build_decoder honor encoder_units/decoder_units/
use_pretrain/use_rf/loss_name/penalty.
"""

from __future__ import annotations

import os
import numpy as np
import pytest
import tensorflow as tf

from resultados.prueba63.src.main_pipeline import ABLATION_CONFIGS, build_ablation_configs
from resultados.prueba63.src.models.models import build_encoder, build_decoder, build_state_projector
from resultados.prueba63.src.pipeline.iteration import Iteration


# --------------------------------------------------------------------------- #
# 1. build_ablation_configs(): exactly 5 correctly parameterized configurations
# --------------------------------------------------------------------------- #

def test_ablation_defines_exactly_five_named_configs():
    names = [c["name"] for c in ABLATION_CONFIGS]
    assert names == ["M1", "M2", "M3", "M4", "M5"]


def test_ablation_configs_returns_five_dicts():
    configs = build_ablation_configs(base_config={"loss_name": "original_mae", "algorithms": "rf_regressor"})
    assert len(configs) == 5
    assert [c["ablation_name"] for c in configs] == ["M1", "M2", "M3", "M4", "M5"]


@pytest.mark.parametrize("name,expected_loss_name,expected_penalty,expected_pretrain,expected_rf", [
    ("M1", "original_mae", [0], False, False),
    ("M2", "pinball_from_penalty", [2], False, False),
    ("M3", "original_mae", [0], True, False),
    ("M4", "pinball_from_penalty", [2], True, False),
    ("M5", "pinball_from_penalty", [0, 2, 4, 6, 8, 10], True, True),
])
def test_ablation_config_flags_match_specification(name, expected_loss_name, expected_penalty,
                                                     expected_pretrain, expected_rf):
    cfg = next(c for c in ABLATION_CONFIGS if c["name"] == name)
    assert cfg["loss_name"] == expected_loss_name
    assert cfg["penalty"] == expected_penalty
    assert cfg["use_pretrain"] == expected_pretrain
    assert cfg["use_rf"] == expected_rf


def test_only_m5_has_random_forest_enabled():
    rf_flags = {c["name"]: c["use_rf"] for c in ABLATION_CONFIGS}
    assert rf_flags == {"M1": False, "M2": False, "M3": False, "M4": False, "M5": True}


def test_ablation_configs_preserve_base_config_keys():
    """loss_name/penalty are specific to each configuration (ABLATION_PROPAGATED_KEYS
    overrides them on purpose); every other base_config key is preserved."""
    base = {"loss_name": "original_mae", "algorithms": "rf_regressor", "contextos": 10}
    configs = build_ablation_configs(base_config=base)
    for cfg in configs:
        assert cfg["algorithms"] == "rf_regressor"
        assert cfg["contextos"] == 10


# --------------------------------------------------------------------------- #
# 2. Iteration() honors encoder_units/decoder_units/use_pretrain/use_rf
# --------------------------------------------------------------------------- #

def _iteration_params(tmp_path, ablation_cfg):
    return {
        "algorithms": "rf_regressor",
        "loss_name": ablation_cfg["loss_name"],
        "penalty": ablation_cfg["penalty"],
        "numero_prueba": f"ablation_{ablation_cfg['name']}",
        "result_path": str(tmp_path),
        "offsets": 1,
        "contextos": 5,
        "steps": 1,
        "seed": 0,
        "lr": 0.001, "ano_test": 2017, "batch_size": 8, "max_epochs": 1,
        "max_epochs_clasificador": 1, "overstep": 0, "coef_de_pond": 0.5,
        "umbrales": 0, "dropout": False, "l2_options": False,
        "use_pretrain": ablation_cfg["use_pretrain"],
        "use_rf": ablation_cfg["use_rf"],
        "encoder_units": ablation_cfg["encoder_units"],
        "decoder_units": ablation_cfg["decoder_units"],
    }


@pytest.mark.parametrize("ablation_cfg", ABLATION_CONFIGS, ids=[c["name"] for c in ABLATION_CONFIGS])
def test_iteration_persists_ablation_flags_in_params(tmp_path, ablation_cfg):
    it = Iteration(_iteration_params(tmp_path, ablation_cfg))
    assert it.p["use_pretrain"] == ablation_cfg["use_pretrain"]
    assert it.p["use_rf"] == ablation_cfg["use_rf"]
    assert it.p["encoder_units"] == ablation_cfg["encoder_units"]
    assert it.p["decoder_units"] == ablation_cfg["decoder_units"]
    assert it.p["loss_name"] == ablation_cfg["loss_name"]
    assert it.p["penalty"] == ablation_cfg["penalty"]


def test_different_ablation_architectures_produce_different_hash_ids(tmp_path):
    """M1 and M2 differ in loss_name/penalty and must therefore cache their
    trained models in different directories."""
    m1 = next(c for c in ABLATION_CONFIGS if c["name"] == "M1")
    m2 = next(c for c in ABLATION_CONFIGS if c["name"] == "M2")

    it1 = Iteration(_iteration_params(tmp_path, m1))
    it2 = Iteration(_iteration_params(tmp_path, m2))

    assert it1.hash_id != it2.hash_id


def test_m1_to_m4_produce_pairwise_distinct_hash_ids(tmp_path):
    """M1-M4 (the configurations retrained by --run_ablation) must each have
    their own hash_id, in particular M1 vs M3 and M2 vs M4, which differ only
    in use_pretrain (included in hash_payload under the "pretrain" key). M5 is
    left out on purpose: it shares M4's hash because use_rf is not part of
    hash_payload (intended: M5 reuses M4's pretrain/finetune directory and
    only adds the RF on top)."""
    m1_to_m4 = [c for c in ABLATION_CONFIGS if c["name"] in ("M1", "M2", "M3", "M4")]
    hashes = {cfg["name"]: Iteration(_iteration_params(tmp_path, cfg)).hash_id for cfg in m1_to_m4}
    assert len(set(hashes.values())) == len(hashes), hashes


def test_default_iteration_is_symmetric_64_units_backward_compatible(tmp_path):
    """Without encoder_units/decoder_units, the default architecture is 64/64 (backward compatible)."""
    params = {
        "algorithms": "rf_regressor", "loss_name": "original_mae",
        "numero_prueba": "legacy", "result_path": str(tmp_path),
        "lr": 0.001, "ano_test": 2017, "batch_size": 8, "max_epochs": 1,
        "max_epochs_clasificador": 1, "overstep": 0, "coef_de_pond": 0.5,
        "umbrales": 0, "dropout": False, "l2_options": False,
    }
    it = Iteration(params)
    assert it.p["encoder_units"] == 64
    assert it.p["decoder_units"] == 64
    assert it.p["use_rf"] is True


# --------------------------------------------------------------------------- #
# 3. Symmetric vs. asymmetric architecture: build_encoder/build_decoder are
#    numerically consistent (working forward pass, no shape errors)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ablation_cfg", ABLATION_CONFIGS, ids=[c["name"] for c in ABLATION_CONFIGS])
def test_ablation_architecture_builds_and_runs_forward_pass(ablation_cfg):
    it_p = {
        "seed": 1, "dropout": False, "l2_options": False,
        "encoder_units": ablation_cfg["encoder_units"],
        "decoder_units": ablation_cfg["decoder_units"],
    }
    contextos, n_enc_feat, n_dec_feat = 6, 3, 5
    encoder_model, _, _, _ = build_encoder((contextos, n_enc_feat), it_p)
    decoder_model = build_decoder(n_dec_feat, it_p)
    projector_model = build_state_projector(ablation_cfg["encoder_units"], ablation_cfg["decoder_units"])

    x_enc = np.random.rand(2, contextos, n_enc_feat).astype("float32")
    x_dec = np.random.rand(2, 1, n_dec_feat).astype("float32")

    state_h, state_c = encoder_model(x_enc)
    assert state_h.shape == (2, ablation_cfg["encoder_units"])

    # The bridge (build_state_projector) maps the encoder state to the decoder
    # width ONCE, before the first step; if the widths already match there is
    # no bridge and the state is used as is.
    if projector_model is not None:
        state_h, state_c = projector_model([state_h, state_c])
    assert state_h.shape == (2, ablation_cfg["decoder_units"])

    out, new_h, new_c = decoder_model([tf.constant(x_dec), state_h, state_c])
    assert out.shape == (2, 1, 1)
    # decoder_model exposes its state in decoder_units on EVERY call: it is a
    # self-contained recurrent step fed back with its own state, without any
    # projection, so the LSTM Constant Error Carousel is preserved.
    assert new_h.shape == (2, ablation_cfg["decoder_units"])
    assert new_c.shape == (2, ablation_cfg["decoder_units"])

    # The returned state must feed a second autoregressive step (regression test
    # of the encoder/decoder shape bug) without going through the projector again.
    out2, new_h2, new_c2 = decoder_model([tf.constant(x_dec), new_h, new_c])
    assert out2.shape == (2, 1, 1)
    assert new_h2.shape == (2, ablation_cfg["decoder_units"])
    assert new_c2.shape == (2, ablation_cfg["decoder_units"])


# --------------------------------------------------------------------------- #
# 4. finetune() with use_pretrain=False trains from scratch and never touches
#    paths.pretrain (regression test: M1 and M3 once produced bit-identical
#    results because finetune() always loaded from paths.pretrain).
# --------------------------------------------------------------------------- #

def _synthetic_windows(n=6, contextos=5, n_enc_feat=3, n_dec_feat=1, offsets=1):
    # x_dec/y carry offsets+1 steps (the full target sequence), see
    # preprocessing.sliding_window.
    n_dec_steps = offsets + 1
    x_enc = np.random.rand(n, contextos, n_enc_feat).astype("float32")
    x_dec = np.random.rand(n, n_dec_steps, n_dec_feat).astype("float32")
    y = np.random.rand(n, n_dec_steps).astype("float32")
    return x_enc, x_dec, y, None


def _scratch_iteration(tmp_path, use_pretrain: bool) -> Iteration:
    offsets = 1
    params = {
        "algorithms": "rf_regressor", "loss_name": "original_mae", "penalty": [0],
        "numero_prueba": f"scratch_pretrain_{use_pretrain}", "result_path": str(tmp_path),
        "offsets": offsets, "contextos": 5, "steps": 1, "seed": 0,
        "lr": 0.001, "ano_test": 2017, "batch_size": 4, "max_epochs": 1,
        "max_epochs_clasificador": 1, "overstep": 0, "coef_de_pond": 0.5,
        "umbrales": 0, "dropout": False, "l2_options": False,
        "use_pretrain": use_pretrain, "use_rf": False,
        "encoder_units": 8, "decoder_units": 8,
    }
    it = Iteration(params)
    x_enc, x_dec, y, dates = _synthetic_windows(offsets=offsets)
    it.train_inputs = (x_enc, x_dec, y, dates)
    it.test_inputs = (x_enc, x_dec, y, dates)
    it._windows_tag = "normal"
    return it


def _scratch_iteration_asymmetric(tmp_path) -> Iteration:
    """Like _scratch_iteration but with encoder_units != decoder_units (64/32),
    the only combination that exercises build_state_projector."""
    offsets = 1
    params = {
        "algorithms": "rf_regressor", "loss_name": "original_mae", "penalty": [0],
        "numero_prueba": "scratch_asymmetric", "result_path": str(tmp_path),
        "offsets": offsets, "contextos": 5, "steps": 1, "seed": 0,
        "lr": 0.001, "ano_test": 2017, "batch_size": 4, "max_epochs": 1,
        "max_epochs_clasificador": 1, "overstep": 0, "coef_de_pond": 0.5,
        "umbrales": 0, "dropout": False, "l2_options": False,
        "use_pretrain": False, "use_rf": False,
        "encoder_units": 64, "decoder_units": 32,
    }
    it = Iteration(params)
    x_enc, x_dec, y, dates = _synthetic_windows(offsets=offsets)
    it.train_inputs = (x_enc, x_dec, y, dates)
    it.test_inputs = (x_enc, x_dec, y, dates)
    it._windows_tag = "normal"
    return it


def test_asymmetric_full_pipeline_train_save_load_predict(tmp_path):
    """End-to-end regression test for encoder_units != decoder_units (64/32).

    it.finetune() builds and trains encoder/decoder/projector from scratch and
    saves them (3 real .keras files); they are then reloaded with
    tf.keras.models.load_model (a real serialization round trip) and
    _predict_iterative is called -- the full life cycle that an integration
    failure between Iteration and build_state_projector would break.
    """
    it = _scratch_iteration_asymmetric(tmp_path)
    it.finetune()

    fin_enc = os.path.join(it.paths.models, "encoder_original_mae_0.keras")
    fin_dec = os.path.join(it.paths.models, "decoder_original_mae_0.keras")
    fin_proj = os.path.join(it.paths.models, "projector_original_mae_0.keras")
    assert os.path.exists(fin_enc) and os.path.exists(fin_dec), "finetune() must save encoder/decoder"
    assert os.path.exists(fin_proj), "finetune() must save the projector when encoder_units != decoder_units"

    enc_model = tf.keras.models.load_model(fin_enc)
    dec_model = tf.keras.models.load_model(fin_dec)
    proj_model = tf.keras.models.load_model(fin_proj)

    x_enc, x_dec, _y, _dates = it.train_inputs
    preds = it._predict_iterative(enc_model, dec_model, x_enc, x_dec, projector_model=proj_model)
    assert preds.shape == (x_enc.shape[0], it.p["offsets"] + 1, 1)
    assert np.isfinite(preds).all()


def test_finetune_scratch_never_touches_pretrain(tmp_path, monkeypatch):
    it = _scratch_iteration(tmp_path, use_pretrain=False)

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("ensure_pretrained() must not be called when use_pretrain=False")
    monkeypatch.setattr(it, "ensure_pretrained", _fail_if_called)

    it.finetune()

    assert os.listdir(it.paths.pretrain) == [], "paths.pretrain must not receive files with use_pretrain=False"
    fin_enc = os.path.join(it.paths.models, "encoder_original_mae_0.keras")
    fin_dec = os.path.join(it.paths.models, "decoder_original_mae_0.keras")
    assert os.path.exists(fin_enc) and os.path.exists(fin_dec), "finetune() must save the model trained from scratch"


# --------------------------------------------------------------------------- #
# 5. Sensitivity of the pinball loss to the tau injected from penalty
# --------------------------------------------------------------------------- #

def test_pinball_penalty_injection_produces_distinct_loss_values():
    """Replicate the tau injection of train_model() (models/training.py).

    tau = (penalty+1)/(penalty+2) is injected into iteration_params['tau'] and
    passed to compute_loss_by_name(). With penalty=0.0 (tau=0.5, median) and
    penalty=10.0 (tau≈0.917, strongly asymmetric) on the SAME y_true/y_pred,
    the two loss values must differ strictly; otherwise the injected tau is
    not reaching the loss (possible silent fallback to the default 0.9).
    """
    from resultados.prueba63.src.models.training import compute_loss_by_name

    y_true = tf.constant([[0.2], [0.5], [0.8], [0.3], [0.6]], dtype=tf.float32)
    y_pred = tf.constant([[[0.5]], [[0.5]], [[0.5]], [[0.5]], [[0.5]]], dtype=tf.float32)

    losses = {}
    for penalty in (0.0, 10.0):
        iteration_params = {"penalty": penalty, "loss_name": "pinball_from_penalty"}
        # Same formula and injection as train_model.
        iteration_params["tau"] = min(0.99, (penalty + 1.0) / (penalty + 2.0))
        losses[penalty] = compute_loss_by_name(
            iteration_params["loss_name"], y_true, y_pred,
            penalty_value=iteration_params["penalty"], tau=iteration_params["tau"],
        )

    loss_p0 = float(losses[0.0].numpy())
    loss_p10 = float(losses[10.0].numpy())
    assert loss_p0 != loss_p10, (
        f"penalty=0.0 (tau=0.5) and penalty=10.0 (tau≈0.917) produced the same "
        f"loss ({loss_p0}) -- the injected tau is not affecting compute_loss_by_name."
    )


def _hash_params(tmp_path, seed: int) -> dict:
    return {
        "algorithms": "rf_regressor", "loss_name": "original_mae", "penalty": [0],
        "numero_prueba": "scratch_seed", "result_path": str(tmp_path),
        "offsets": 1, "contextos": 5, "steps": 1, "seed": seed,
        "lr": 0.001, "ano_test": 2017, "batch_size": 4, "max_epochs": 1,
        "max_epochs_clasificador": 1, "overstep": 0, "coef_de_pond": 0.5,
        "umbrales": 0, "dropout": False, "l2_options": False,
        "use_pretrain": False, "use_rf": False,
        "encoder_units": 32, "decoder_units": 32,
    }


def test_seed_isolates_cache_hash_only_when_it_differs_from_published(tmp_path):
    """Per-seed cache isolation.

    The published seed does NOT enter the hash (validated weights keep their
    directory byte for byte), but any other seed DOES, so that a sensitivity
    experiment neither silently resumes from nor overwrites the .keras files of
    the published run.
    """
    from resultados.prueba63.src.pipeline.iteration import _PUBLISHED_RUN_SEED

    hash_published = Iteration(_hash_params(tmp_path, _PUBLISHED_RUN_SEED)).hash_id
    assert Iteration(_hash_params(tmp_path, _PUBLISHED_RUN_SEED)).hash_id == hash_published

    for other in (42, 123):
        assert Iteration(_hash_params(tmp_path, other)).hash_id != hash_published

    assert (Iteration(_hash_params(tmp_path, 42)).hash_id
            != Iteration(_hash_params(tmp_path, 123)).hash_id)
