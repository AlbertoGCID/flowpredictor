"""Experiment configuration layer: CV plan, ablation configurations, baselines and fold metrics.

Also contains the legacy grid-search driver :func:`run_experiments`; the
published experiments are orchestrated by ``run_experiments_phase3.py``.
"""
from __future__ import annotations

import copy
import itertools
import json
import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..log.log_config import init_logger,get_logger
from .config_schema import CONFIG, AblationVariant
from .evaluation.evaluator import generate_global_report,plot_flow_rain_from_csv
from .evaluation.metrics import (
    compute_extreme_metrics, peak_timing_error, nse, kge, bootstrap_ci_standard_metrics,
)
from .models.baselines import (
    QuantileRegressionBaseline, GradientBoostingQuantileBaseline, HeuristicEnsembleBaseline,
)
from .pipeline.utils import sanitize_for_log
from .pipeline.iteration import Iteration

# ---------------- Grid & CV helpers ----------------
def expand_grid(config: Dict, outer_keys: List[str]) -> Iterable[Dict]:
    """Yield every combination of the values of ``outer_keys`` (Cartesian product).

    Args:
        config (Dict): Configuration; scalar values count as one-element lists.
        outer_keys (List[str]): Keys to expand.

    Yields:
        Dict: One ``{key: value}`` combination.
    """
    lists = []
    for k in outer_keys:
        v = config.get(k, [])
        v = v if isinstance(v, (list, tuple)) else [v]
        lists.append((k, v))
    keys = [k for k, _ in lists]
    vals = [v for _, v in lists]
    for combo in itertools.product(*vals):
        d = {k: combo[i] for i, k in enumerate(keys)}
        yield d

def iter_test_years(crossvalidation: bool, base_config: Dict) -> List[int]:
    """Test years of a legacy run.

    Args:
        crossvalidation (bool): Use the fixed 2014-2022 range.
        base_config (Dict): Configuration with ``ano_test``.

    Returns:
        List[int]: Test years.
    """
    if crossvalidation:
        return list(range(2014, 2023))
    years = base_config.get("ano_test", [])
    years = years if isinstance(years, (list, tuple)) else [years]
    return [int(y) for y in years]


# ---------------- Expanding-window CV + 2025 holdout ----------------

# Values from experiments_config.toml [cv]. 2014-2016 are warm-up years inside
# the training set of the expanding folds, not test years of their own: before
# 2017 there is not enough history for the classifier's internal OOF loop
# (self.p["burnin_years"]) to have any internal year available. LOYO has no
# such restriction (each test year trains on every other available
# hydrological year, past and future); its actual upper bound depends on which
# bundles exist (see run_experiments_phase3.py::_bundle_ready).
EXPANDING_CV_YEARS: List[int] = list(CONFIG.cv["expanding_cv_years"])
HOLDOUT_YEAR: int = int(CONFIG.cv["holdout_year"])
LOYO_CV_YEARS: List[int] = list(CONFIG.cv["loyo_cv_years"])


def build_expanding_window_cv_plan(
    cv_years: List[int] = EXPANDING_CV_YEARS,
    holdout_year: int = HOLDOUT_YEAR,
    dataset_start_year: int = int(CONFIG.cv["dataset_start_year"]),
) -> List[Dict]:
    """Expanding-window temporal validation plan.

    - One fold per year in ``cv_years``, with Train=[dataset_start_year, T-1], Test=T.
    - A final fold flagged ``is_holdout=True`` for ``holdout_year``, with
      Train=[dataset_start_year, holdout_year-1], Test=holdout_year.

    Runs nothing: it describes which (test year, ``split="JunioExpanding"``)
    combinations must be evaluated. The corresponding bundles are built by
    ``dataset_generator.run_expanding_window()``.

    Args:
        cv_years (List[int]): Cross-validation test years.
        holdout_year (int): Holdout label year.
        dataset_start_year (int): First year of the dataset.

    Returns:
        List[Dict]: Folds with ``test_year``, ``train_years`` and ``is_holdout``.
    """
    plan = []
    for test_year in cv_years:
        plan.append({
            "test_year": int(test_year),
            "train_years": list(range(dataset_start_year, test_year)),
            "is_holdout": False,
        })
    plan.append({
        "test_year": int(holdout_year),
        "train_years": list(range(dataset_start_year, holdout_year)),
        "is_holdout": True,
    })
    return plan


def log_extended_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_label: str,
    threshold_p90: Optional[float] = None,
    out_dir: Optional[str] = None,
    n_bootstraps: int = 200,
) -> Dict:
    """Compute the full metric set of a fold (NSE, KGE, extreme-event metrics, peak timing, bootstrap CIs).

    Args:
        y_true (np.ndarray): Observed inflow.
        y_pred (np.ndarray): Predicted inflow.
        fold_label (str): Fold label.
        threshold_p90 (Optional[float]): Extreme-event threshold fixed on TRAIN.
        out_dir (Optional[str]): If given, the result is saved as
            ``<out_dir>/metrics_<fold_label>.json``.
        n_bootstraps (int): Bootstrap resamples for the 95% CIs.

    Returns:
        Dict: Metrics, ``PeakTimingError`` block and ``confidence_intervals_95``.
    """
    extreme = compute_extreme_metrics(y_true, y_pred, threshold_p90=threshold_p90)
    timing = peak_timing_error(y_true, y_pred, threshold_p90=threshold_p90)
    ci = bootstrap_ci_standard_metrics(y_true, y_pred, threshold_p90=threshold_p90, n_bootstraps=n_bootstraps)

    result = {
        "fold": fold_label,
        "NSE": nse(y_true, y_pred),
        "KGE": kge(y_true, y_pred),
        "HitRatio": extreme["HitRatio"],
        "FARate": extreme["FARate"],
        "FARatio": extreme["FARatio"],
        "F1": extreme["F1"],
        "Precision": extreme["Precision"],
        "Recall": extreme["Recall"],
        "n_extreme": extreme["n_extreme"],
        "PeakTimingError": timing,
        "confidence_intervals_95": ci,
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"metrics_{fold_label}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=float)

    return result


# ---------------- Ablation study (M1-M5) ----------------
# The ablation axes are pretraining, loss function (the asymmetric loss is the
# central contribution) and the RF meta-learner:
# M1: no pretraining, symmetric MAE, no RF     -- baseline without asymmetry
# M2: no pretraining, pinball loss, no RF      -- effect of the asymmetry alone
# M3: pretraining,    symmetric MAE, no RF     -- effect of pretraining alone
# M4: pretraining,    pinball loss, no RF      -- everything but the RF meta-learner
# M5: pretraining,    pinball loss, RF         -- proposed system (Flowpredictor)

# Values from experiments_config.toml [ablation_variants.*]. penalty is
# converted back to a list (not a tuple), the type consumed by
# run_experiments_phase3.py and Iteration.
ABLATION_CONFIGS: List[Dict] = [
    {
        "name": name,
        "description": variant.description,
        "use_pretrain": variant.use_pretrain,
        "use_rf": variant.use_rf,
        "loss_name": variant.loss_name,
        "penalty": list(variant.penalty),
        "encoder_units": variant.encoder_units,
        "decoder_units": variant.decoder_units,
    }
    for name, variant in CONFIG.variants.items()
]

# Derived from the AblationVariant fields (except "description", which is not
# passed to Iteration), so the propagated keys can never diverge from the schema.
ABLATION_PROPAGATED_KEYS: List[str] = [
    f for f in AblationVariant.__dataclass_fields__ if f != "description"
]

# Guard: a key present in ABLATION_CONFIGS but not propagated to Iteration
# would fail silently, producing configurations that differ only on paper.
# Any mismatch fails loudly at import time instead.
_expected_ablation_keys = {"name", "description", *ABLATION_PROPAGATED_KEYS}
for _cfg in ABLATION_CONFIGS:
    if set(_cfg) != _expected_ablation_keys:
        raise AssertionError(
            f"ABLATION_CONFIGS['{_cfg['name']}'] has keys {sorted(_cfg)} but "
            f"ABLATION_PROPAGATED_KEYS expects {sorted(_expected_ablation_keys)}."
        )

# Guard (encoder -> decoder Dense bridge): the five configurations must be
# symmetric (encoder_units == decoder_units) so that the state projector is
# never built for M1-M5. The projector stays in the code for asymmetric widths
# outside ABLATION_CONFIGS, but must not be silently reintroduced here by a
# TOML change.
for _cfg in ABLATION_CONFIGS:
    if _cfg["encoder_units"] != _cfg["decoder_units"]:
        raise AssertionError(
            f"ABLATION_CONFIGS['{_cfg['name']}'] is asymmetric "
            f"(encoder_units={_cfg['encoder_units']} != "
            f"decoder_units={_cfg['decoder_units']}); M1-M5 must remain "
            f"symmetric (no Dense bridge in the recurrent path)."
        )


def build_ablation_configs(base_config: Optional[Dict] = None) -> List[Dict]:
    """Build the five ablation configurations (M1..M5) as Iteration parameter overrides.

    No training is launched; each resulting dict is ready for
    ``Iteration(params)``. Iteration supplies its own defaults for any key not
    set here.

    Args:
        base_config (Optional[Dict]): Base parameters (empty dict if ``None``).

    Returns:
        List[Dict]: One parameter dict per configuration.
    """
    base = copy.deepcopy(base_config) if base_config is not None else {}
    configs = []
    for ablation in ABLATION_CONFIGS:
        cfg = copy.deepcopy(base)
        cfg["ablation_name"] = ablation["name"]
        cfg["ablation_description"] = ablation["description"]
        for key in ABLATION_PROPAGATED_KEYS:
            cfg[key] = copy.deepcopy(ablation[key])
        configs.append(cfg)
    return configs


# ---------------- Baselines (models/baselines.py) ----------------

# Names without a suffix are those of the published run (tau=0.50) and are NOT
# renamed: they are the 'config' key of consolidated rows and manuscript
# tables. The tau=0.90 variants are separate configurations.
# API note: in GradientBoostingRegressor(loss="quantile") the quantile is
# `alpha` (the wrapper translates it); in QuantileRegressor the quantile is
# `quantile` and `alpha` is the L1 regularization -- do not confuse them.
BASELINE_REGISTRY = {
    "quantile_regression": lambda: QuantileRegressionBaseline(quantile=0.5),
    "gradient_boosting_quantile": lambda: GradientBoostingQuantileBaseline(quantile=0.5, n_estimators=200),
    "quantile_regression_tau90": lambda: QuantileRegressionBaseline(quantile=0.9),
    "gradient_boosting_tau90": lambda: GradientBoostingQuantileBaseline(quantile=0.9, n_estimators=200),
    "heuristic_ensemble": lambda: HeuristicEnsembleBaseline(mode="weighted"),
}

# sklearn baselines evaluated in each fold. Extending this tuple (not the
# registry) is what makes a variant enter results_consolidated.csv.
SKLEARN_BASELINES = ("quantile_regression", "gradient_boosting_quantile",
                     "quantile_regression_tau90", "gradient_boosting_tau90")


def run_baselines_for_iteration(it: Iteration, threshold_p90: Optional[float] = None) -> Dict[str, Dict]:
    """Train and evaluate the sklearn baselines on the Seq2Seq windows.

    The baselines use exactly the same leak-free windows as the Seq2Seq + RF
    pipeline (see tests/test_leakage.py), for a like-for-like comparison.
    Quantile regression and quantile gradient boosting are trained on the
    flattened (encoder + decoder) inputs; the heuristic ensemble needs base
    model predictions and is evaluated separately. Requires
    ``it.ensure_and_load_data()`` to have been called.

    Args:
        it (Iteration): Iteration with data loaded.
        threshold_p90 (Optional[float]): Extreme-event threshold fixed on TRAIN.

    Returns:
        Dict[str, Dict]: Metrics per baseline name.
    """
    (x_tr_enc, x_tr_dec, y_tr, _), (x_te_enc, x_te_dec, y_te, te_dates) = it.build_windows_normal()

    # The baselines take ONE 2D/3D matrix, not a Keras-style [encoder, decoder]
    # list: flattened encoder and decoder inputs are concatenated.
    x_tr_combined = np.concatenate(
        [x_tr_enc.reshape(x_tr_enc.shape[0], -1), x_tr_dec.reshape(x_tr_dec.shape[0], -1)], axis=1
    )
    x_te_combined = np.concatenate(
        [x_te_enc.reshape(x_te_enc.shape[0], -1), x_te_dec.reshape(x_te_dec.shape[0], -1)], axis=1
    )

    results: Dict[str, Dict] = {}
    for name in SKLEARN_BASELINES:
        model = BASELINE_REGISTRY[name]()
        # y_tr/y_te are the full sequence [Qe(t+1)...Qe(t+offsets+1)]; these
        # baselines train one sub-model per horizon column, so fit() uses all of
        # them, but evaluation only uses the final evaluated horizon (the last
        # step), as in Iteration.ensure_predictions (step_idx).
        model.fit(x_tr_combined, y_tr)
        y_pred_norm = model.predict(x_te_combined)[:, -1]
        y_pred = it._denorm_qe(y_pred_norm)
        y_true = it._denorm_qe(y_te[:, -1])
        results[name] = log_extended_fold_metrics(y_true, y_pred, fold_label=f"baseline_{name}", threshold_p90=threshold_p90)

    return results

# ---------------- Legacy grid-search orchestration ----------------

def _as_list(x: Any) -> List[Any]:
    """Wrap a scalar into a list; convert tuples to lists."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def _expand_outer_grid(base_cfg: dict, outer_keys: List[str], crossvalidation: bool) -> List[Dict[str, Any]]:
    """Expand the outer grid into one parameter dict per iteration.

    Combines the values of ``outer_keys`` and, if ``crossvalidation``, the
    years 2014-2022. ``penalty`` is also propagated to ``pretrain_penalties``
    for ``ensure_pretrained()``.

    Args:
        base_cfg (dict): Base configuration.
        outer_keys (List[str]): Keys to expand.
        crossvalidation (bool): Iterate over the fixed year range.

    Returns:
        List[Dict[str, Any]]: Iteration parameters.
    """
    cfg = copy.deepcopy(base_cfg)
    # values of the outer keys (taken as-is from the base config if absent)
    key_values = {}
    for k in outer_keys:
        if k in cfg:
            key_values[k] = _as_list(cfg[k])
        else:
            key_values[k] = _as_list(cfg.get(k, [None]))

    # years
    if crossvalidation:
        years = list(range(2014, 2023))
    else:
        years = _as_list(cfg.get("ano_test", [])) or _as_list(base_cfg.get("ano_test", []))

    combos = []
    for values in itertools.product(*[key_values[k] for k in outer_keys]):
        combo = copy.deepcopy(cfg)
        for k, v in zip(outer_keys, values):
            combo[k] = v
        # make sure pretraining penalties are set
        if "penalty" in cfg:
            combo["pretrain_penalties"] = cfg["penalty"]

        for year in years:
            it_params = copy.deepcopy(combo)
            it_params["ano_test"] = int(year)
            combos.append(it_params)
    return combos

def run_experiments(configuracion: dict, outer_keys: List[str], crossvalidation: bool) -> None:
    """Run every combination of the outer grid (and years, if ``crossvalidation``).

    Args:
        configuracion (dict): Configuration.
        outer_keys (List[str]): Keys to expand.
        crossvalidation (bool): Iterate over the fixed year range.
    """
    logger = get_logger()
    all_iters = _expand_outer_grid(configuracion, outer_keys, crossvalidation)
    logger.info("%d iterations will run (outer grid%s).",
                len(all_iters),
                " + crossvalidation" if crossvalidation else "")

    for idx, params in enumerate(all_iters, start=1):
        try:
            it = Iteration(params)

            # Ablation switch: pretraining
            use_pretrain = it.p.get("use_pretrain", True)

            logger.info("── Iteration %d/%d | hash=%s | split=%s | year=%s | alg=%s | pretrain=%s",
                        idx, len(all_iters), it.hash_id,
                        it.p.get("split"), it.p.get("ano_test"),
                        it.p.get("algorithms"), use_pretrain)

            # 1) Data
            it.ensure_and_load_data()

            # 2-3) Conditional pretraining (ablation study)
            if use_pretrain:
                it.build_windows_qe_input0()
                it.ensure_pretrained()
            else:
                logger.info("--- ABLATION STUDY: skipping masked pretraining ---")

            # 4) Normal windows, so that the cache is ready
            it.build_windows_normal()

            # 5) Base predictions (trained from scratch if use_pretrain=False)
            it.ensure_predictions(force=False)

            # 6) Operational metrics (false alarms, timing error)
            summary_df = it.compute_metrics_for_all_models(metrics=[
                "overall_hydroeval",
                "top10_hits_misses",
                "alarm_metrics",  # false-alarm metrics
                "timing_error"    # lead/lag metrics
            ], force=False)
            logger.info("Metrics summary:\n%s", summary_df)

            # Ablation switch: Random Forest meta-learner
            use_rf = it.p.get("use_rf", True)
            if use_rf:
                it.ensure_ensemble_labels(subset="train", force=False)
                it.ensure_ensemble_labels(subset="test", force=False)

                # 7) Train the classifier
                it.train_classifier(force=False)

                # 8) Classifier predictions on TEST + integrated metrics
                it.predict_classifier(force=False)
                it.evaluate_classifier(metrics=[
                    "overall_hydroeval",
                    "top10_hits_misses",
                    "alarm_metrics",
                    "timing_error"
                ], force=False)

                # NSE/KGE/HitRatio/FARate/FARatio/F1/PeakTimingError + 95% CI
                algo = str(it.p.get("algorithms", "rf_regressor"))
                cls_npz = os.path.join(it.paths.predictions_cache, f"classifier_{algo}.npz")
                if os.path.exists(cls_npz):
                    data = np.load(cls_npz, allow_pickle=True)
                    fold_label = f"{it.p.get('ano_test')}_{it.p.get('ablation_name', 'default')}"
                    log_extended_fold_metrics(
                        data["y_true"].astype(float), data["y_pred"].astype(float),
                        fold_label=fold_label, out_dir=it.paths.predictions_cache,
                    )
            else:
                logger.info("--- ABLATION STUDY: skipping the Random Forest meta-learner (use_rf=False) ---")

            logger.info("✅ Iteration %d/%d completed (hash=%s).", idx, len(all_iters), it.hash_id)

        except Exception as e:
            logger.exception("❌ Iteration %d/%d failed: %s", idx, len(all_iters), str(e))
            continue


if __name__ == "__main__":
    import json as _json
    log_level_setup = logging.INFO
    # Legacy direct invocation: builds its own configuration from scratch;
    # Iteration supplies defaults for any key not set below.
    configuracion: Dict = {}
    configuracion['offsets'] = [0,1,2,3]
    configuracion["algorithms"] = ["rf_regressor"]
    configuracion['penalty'] = [0, 2, 4, 6, 8, 10]
    configuracion["l2_options"] = [True, False]
    configuracion["dropout"] = [True, False]
    configuracion['lr'] = [0.002,0.005]
    configuracion["ano_test"] = [2017]
    configuracion["numero_prueba"] = "63"
    configuracion["max_epochs"] = [100]
    configuracion["max_epochs_clasificador"] = [150]
    configuracion["seed"] = 48
    configuracion["k_high"]= [0.0,2.0]
    configuracion["eval_threshold"] = ["p90"]
    configuracion["use_pretrain"] = [False]
    crossvalidation = True
    # l2_options/dropout are used by models/models.py and hashed in
    # pipeline/iteration.py.
    outer_keys = ["l2_options", "lr", "dropout"]

    # Logging
    cfg_log = sanitize_for_log(configuracion)
    init_logger(log_level_setup, numero_prueba=configuracion["numero_prueba"])
    logger = get_logger()

    logger.info("Running the experiment with the following configuration:\n%s",
                _json.dumps(cfg_log, indent=2, ensure_ascii=False))

    # Launch the experiment(s)
    run_experiments(cfg_log, outer_keys, crossvalidation)
    generate_global_report(numero_prueba=cfg_log["numero_prueba"], make_plots=True, over=True)
    res = plot_flow_rain_from_csv()