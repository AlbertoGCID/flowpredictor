#!/usr/bin/env python3
"""Command-line orchestrator of the Flowpredictor experiments (Phase 3).

Run as a module from the directory that contains ``flowpredictor/``, with
``flowpredictor/`` on ``PYTHONPATH`` (imports are absolute,
``from resultados.prueba63...``)::

    PYTHONPATH=flowpredictor python3 -m resultados.prueba63.src.run_experiments_phase3 --run_cv

Main flags:
  --run_cv          M5 (full framework) plus the tabular baselines of
                    ``models/baselines.py`` on the expanding-window CV folds
                    (test years 2017-2023; the 2025 holdout is excluded, see
                    ``build_expanding_window_cv_plan``).
  --run_ablation    M1-M4 on the same years.
  --run_robustness  Monte Carlo rainfall perturbation (uniform +/-10%,
                    Gaussian +/-5%, 50 replicas) on the selected model.
  --all             The three above.

Incremental checkpointing and automatic resume (see ``_save_results`` and
``_is_fold_complete``): each fold is written to ``results_consolidated.csv``
(through a temporary file and ``os.replace``, which is atomic) as soon as it is
computed, and every phase skips folds that are already complete in that CSV,
so an interruption in the middle of ``--all`` never forces recomputation. Use
``--no_resume`` to recompute everything anyway.
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from resultados.prueba63.log.log_config import init_logger, get_logger
from resultados.prueba63.src.config import CONFIG
from resultados.prueba63.src.pipeline import dataio
from resultados.prueba63.src.pipeline.iteration import Iteration, _canonical_base
from resultados.prueba63.src.pipeline.preprocessing import prepare_model_inputs
from resultados.prueba63.src.data.dataset_generator import normalize_with_params
from resultados.prueba63.src.models.baselines import HeuristicEnsembleBaseline
from resultados.prueba63.src.evaluation.model_selection import build_final_report, run_model_selection
from resultados.prueba63.src.main_pipeline import (
    ABLATION_CONFIGS,
    ABLATION_PROPAGATED_KEYS,
    HOLDOUT_YEAR,
    LOYO_CV_YEARS,
    SKLEARN_BASELINES,
    build_expanding_window_cv_plan,
    log_extended_fold_metrics,
    run_baselines_for_iteration,
)
from resultados.prueba63.src.evaluation.perturbation import (
    run_perturbation_test, perturb_rainfall_uniform, perturb_rainfall_gaussian,
)
from resultados.prueba63.src.evaluation.metrics import compute_extreme_metrics

# Paths, schema columns and base hyperparameters come from
# experiments_config.toml. Importing this module does not mutate CONFIG: the
# absolute paths are resolved explicitly by PipelineConfig.from_toml from the
# location of the TOML file itself.
NUMERO_PRUEBA = str(CONFIG.meta["numero_prueba"])
SPLIT_LABEL = str(CONFIG.cv["split_label"])  # bundles built by dataset_generator.run_expanding_window()
# True leave-one-year-out: training uses years both before AND after the test
# year (see load_and_split_data(expanding_window=False)); bundles built by
# dataset_generator.run(). Same hydrological-year convention (July -> June) as
# SPLIT_LABEL, so that the M1-M5 comparison never mixes different date
# semantics between the expanding and LOYO strategies.
LOYO_SPLIT_LABEL = str(CONFIG.cv["loyo_split_label"])

RESULTS_CSV = Path(CONFIG.paths["results_csv"])
PREDICTIONS_DIR = Path(CONFIG.paths["predictions_dir"])

# Columns a row must have to be considered "complete" when resuming. The
# primary key of each row is 'fold' (= config + "_" + test year, e.g.
# "M5_2019" or "robustness_uniform_2025"); tests/test_results_integrity.py
# imports these same constants so it never diverges from the resume criterion.
METRIC_COLUMNS = list(CONFIG.schema["metric_columns"])
ROBUSTNESS_COLUMNS = list(CONFIG.schema["robustness_columns"])

# Multi-horizon: canonical mapping offset -> evaluated forecast horizon (see
# preprocessing.py::sliding_window -- step 0=24h, step 1=48h, step 2=72h;
# Iteration.ensure_predictions always evaluates step self.p["offsets"], the
# LAST step of the decoder sequence, not the first).
HORIZON_LABELS = {0: "24h", 1: "48h", 2: "72h"}


def _current_horizon_tag() -> str:
    """Horizon label of the current run, derived from ``BASE_ITERATION_DEFAULTS``.

    Single source of truth: ``--offset`` overrides it in :func:`main` before
    any dispatch, and no ``ABLATION_PROPAGATED_KEYS`` entry includes
    ``offsets``, so it is identical for M1-M5 and the baselines within a run.
    ``"48h"`` is the historical value and adds no fold suffix, so that the
    already-validated rows of the CSV are unaffected.

    Returns:
        str: ``"24h"``, ``"48h"``, ``"72h"`` or ``"offset<N>"``.
    """
    return HORIZON_LABELS.get(int(BASE_ITERATION_DEFAULTS["offsets"]),
                               f"offset{BASE_ITERATION_DEFAULTS['offsets']}")


# Published variant: seed 92 and p90 threshold. Both are literals on purpose
# (not CONFIG), for the same reason as _PUBLISHED_RUN_SEED in pipeline/iteration.py.
PUBLISHED_SEED = 92
PUBLISHED_THRESHOLD_PCT = 90


def _current_seed() -> int:
    """Training seed of the current run.

    Returns:
        int: ``BASE_ITERATION_DEFAULTS["seed"]``.
    """
    return int(BASE_ITERATION_DEFAULTS["seed"])


def _current_threshold_pct() -> int:
    """Percentile of the extreme-event threshold of the current run.

    Derived from ``eval_threshold`` (``"p90"``/``"p95"``), the same key that
    :meth:`Iteration.get_train_threshold` reads from ``norm_params["Qe"]``,
    already precomputed in every bundle.

    Returns:
        int: The percentile (e.g. ``90``).
    """
    raw = BASE_ITERATION_DEFAULTS["eval_threshold"]
    raw = raw[0] if isinstance(raw, (list, tuple)) else raw
    return int(str(raw).lstrip("p"))


def _apply_variant_suffix(fold_label: str, suffix: str) -> str:
    """Insert a variant suffix into a fold label.

    The suffix goes in the SAME position used by the training path (before the
    year: ``'M2_2017' -> 'M2_p95_2017'``), so that a variant has a single label
    wherever it comes from. Labels without a trailing year (e.g.
    ``'M2_holdout_final'``) get it at the end.

    Args:
        fold_label (str): Original fold label.
        suffix (str): Variant suffix (may be empty).

    Returns:
        str: The suffixed label.
    """
    if not suffix:
        return fold_label
    return re.sub(r"_(\d{4})$", rf"{suffix}_\1", fold_label) if re.search(r"_\d{4}$", fold_label) \
        else f"{fold_label}{suffix}"


def _variant_suffix() -> str:
    """Additive experimental-variant suffix for fold labels.

    Same criterion as the horizon suffix: the published variant (seed 92 +
    p90) adds nothing, so already-consolidated rows and ``.npz`` files keep
    their labels byte for byte; only deviations are marked. Without it, a run
    with another seed would reuse the existing label and ``_save_results``
    would OVERWRITE it (upsert by ``fold``).

    Returns:
        str: ``""``, ``"_s<seed>"``, ``"_p<pct>"`` or both.
    """
    seed, pct = _current_seed(), _current_threshold_pct()
    return ("" if seed == PUBLISHED_SEED else f"_s{seed}") + \
           ("" if pct == PUBLISHED_THRESHOLD_PCT else f"_p{pct}")


BASE_ITERATION_DEFAULTS: Dict[str, Any] = dict(
    algorithms=CONFIG.base["algorithms"],
    loss_name=list(CONFIG.base["loss_name"]),
    penalty=list(CONFIG.base["penalty"]),
    contextos=CONFIG.base["contextos"],
    offsets=CONFIG.base["offsets"],
    steps=CONFIG.base["steps"],
    overstep=CONFIG.base["overstep"],
    batch_size=CONFIG.base["batch_size"],
    max_epochs=CONFIG.base["max_epochs"],
    max_epochs_clasificador=CONFIG.base["max_epochs_clasificador"],
    # Early stopping on train_loss (the pipeline has no validation split, see
    # models/training.py): avoids exhausting all epochs on loss/penalty
    # combinations that have already converged. Opt-in through Iteration; with
    # patience=None the full max_epochs are always run.
    early_stopping_patience=CONFIG.base["early_stopping_patience"],
    early_stopping_min_delta=CONFIG.base["early_stopping_min_delta"],
    lr=CONFIG.base["lr"],
    dropout=CONFIG.base["dropout"],
    l2_options=CONFIG.base["l2_options"],
    coef_de_pond=CONFIG.base["coef_de_pond"],
    umbrales=CONFIG.base["umbrales"],
    eval_threshold=CONFIG.base["eval_threshold"],
    split=SPLIT_LABEL,
    numero_prueba=NUMERO_PRUEBA,
    seed=CONFIG.base["seed"],
)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _bundle_ready(test_year: int, split_label: str = SPLIT_LABEL) -> bool:
    """Whether the normalized data bundle of a test year exists.

    Args:
        test_year (int): Test year.
        split_label (str): Bundle split label.

    Returns:
        bool: ``True`` if the bundle has its normalized partitions.
    """
    base = dataio.resolve_bundle_base(CONFIG, split_label, test_year)
    return dataio.bundle_has_normalized(base)


def _load_existing_results() -> Optional[pd.DataFrame]:
    """Load the consolidated results CSV.

    Returns:
        Optional[pd.DataFrame]: The results, or ``None`` if the CSV does not
        exist, is empty, or cannot be read (e.g. truncated by an interruption
        during a non-atomic write of an older version of this script).
    """
    if not RESULTS_CSV.exists():
        return None
    try:
        df = pd.read_csv(RESULTS_CSV)
    except Exception:
        return None
    return df if not df.empty else None


def _is_fold_complete(existing_df: Optional[pd.DataFrame], fold_label: str,
                       required_cols: List[str] = METRIC_COLUMNS) -> bool:
    """Whether a fold is already present with no missing required metric.

    Args:
        existing_df (Optional[pd.DataFrame]): Existing results.
        fold_label (str): Fold primary key.
        required_cols (List[str]): Columns that must be non-NaN.

    Returns:
        bool: ``True`` if ``fold_label`` is in ``existing_df`` and none of
        ``required_cols`` is NaN.
    """
    if existing_df is None or "fold" not in existing_df.columns:
        return False
    match = existing_df.loc[existing_df["fold"] == fold_label]
    if match.empty:
        return False
    present = [c for c in required_cols if c in existing_df.columns]
    if not present:
        return False
    return bool(not match.iloc[0][present].isna().any())


def _save_results(row_or_rows: Union[Dict[str, Any], Sequence[Optional[Dict[str, Any]]]],
                  logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Atomic incremental checkpoint of one or more result rows.

    Called immediately after evaluating EACH individual fold (not at the end
    of a phase). Rows are upserted by ``fold`` (primary key = config + test
    year), so recomputing a fold replaces it instead of duplicating it, and
    the CSV is written through a temporary file and ``os.replace()`` so that
    an interruption never leaves ``results_consolidated.csv`` truncated.

    Args:
        row_or_rows (Union[Dict[str, Any], Sequence[Optional[Dict[str, Any]]]]):
            A row or a list of rows (``None`` entries are ignored).
        logger (logging.Logger): Logger.

    Returns:
        Optional[pd.DataFrame]: The combined results, or ``None`` if there
        was nothing to save.
    """
    rows = row_or_rows if isinstance(row_or_rows, list) else [row_or_rows]
    rows = [r for r in rows if r is not None]
    if not rows:
        return None

    new_df = pd.DataFrame(rows)

    # Variant stamp applied in a single place (instead of in each of the
    # places where a row is built): every row is tagged with the seed and the
    # threshold percentile that produced it, which is what later allows the
    # sensitivity experiments to be grouped. Pre-existing rows are filled with
    # the published variant (92 / p90), which is the one that produced them.
    new_df["seed"] = _current_seed()
    new_df["threshold_percentile"] = _current_threshold_pct()

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_results()
    if existing is not None and "fold" in existing.columns:
        for col, default in (("seed", PUBLISHED_SEED), ("threshold_percentile", PUBLISHED_THRESHOLD_PCT)):
            if col not in existing.columns:
                existing[col] = default
            else:
                existing[col] = existing[col].fillna(default)
        existing = existing.loc[~existing["fold"].isin(new_df["fold"])]
    else:
        existing = pd.DataFrame()

    combined = pd.concat([existing, new_df], ignore_index=True)

    tmp_path = RESULTS_CSV.with_suffix(RESULTS_CSV.suffix + ".tmp")
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, RESULTS_CSV)  # atomic on POSIX: never leaves a half-written CSV

    logger.info("[checkpoint] %s -> %s (%d total rows)",
                ", ".join(str(r["fold"]) for r in rows), RESULTS_CSV.name, len(combined))
    return combined


def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested metric dictionary into CSV columns.

    Args:
        metrics (Dict[str, Any]): Output of ``log_extended_fold_metrics``.

    Returns:
        Dict[str, Any]: Flat metric columns, including peak-timing fields and
        bootstrap 95% confidence bounds.
    """
    flat = {
        "NSE": metrics.get("NSE"),
        "KGE": metrics.get("KGE"),
        "HitRatio": metrics.get("HitRatio"),
        "FARate": metrics.get("FARate"),
        "FARatio": metrics.get("FARatio"),
        "F1": metrics.get("F1"),
        "Precision": metrics.get("Precision"),
        "Recall": metrics.get("Recall"),
        "n_extreme": metrics.get("n_extreme"),
    }
    timing = metrics.get("PeakTimingError") or {}
    flat["PeakTiming_mean_lag"] = timing.get("mean_lag")
    flat["PeakTiming_mean_absolute_lag"] = timing.get("mean_absolute_lag")
    flat["PeakTiming_n_peaks"] = timing.get("n_peaks")

    for metric_name, ci in (metrics.get("confidence_intervals_95") or {}).items():
        flat[f"{metric_name}_CI_lower"] = ci.get("lower")
        flat[f"{metric_name}_CI_upper"] = ci.get("upper")
    return flat


def _save_predictions(fold_label: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save the observed and predicted series of a fold as ``<fold>.npz``.

    Args:
        fold_label (str): Fold label (file name).
        y_true (np.ndarray): Observed inflow.
        y_pred (np.ndarray): Predicted inflow.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREDICTIONS_DIR / f"{fold_label}.npz", y_true=y_true, y_pred=y_pred)


def _train_threshold_p90(it: Iteration, fold_label: str, logger: logging.Logger) -> float:
    """Extreme-event threshold fixed on the TRAINING partition.

    Uses :meth:`Iteration.get_train_threshold`, which honors
    ``self.p["eval_threshold"]`` and reads the percentile precomputed without
    imputed rows (see
    ``dataset_generator.calculate_normalization_params_excl_imputed``). It is
    passed explicitly to ``log_extended_fold_metrics`` so that
    ``compute_extreme_metrics``/``peak_timing_error`` never recompute it on the
    TEST ``y_true`` (data leakage).

    Args:
        it (Iteration): Iteration with its data loaded.
        fold_label (str): Fold label (for logging).
        logger (logging.Logger): Logger.

    Returns:
        float: The threshold in physical units.
    """
    thr = it.get_train_threshold()
    logger.info("[%s] threshold_p90 (train) = %.6f", fold_label, thr)
    return thr


# --------------------------------------------------------------------------- #
# --run_cv (M5 + baselines) / --run_ablation (M1-M4)
# --------------------------------------------------------------------------- #

def _run_single_config(it_params: Dict[str, Any], fold_label: str, logger: logging.Logger,
                        existing_df: Optional[pd.DataFrame] = None,
                        strategy: str = "expanding",
                        split_label: str = SPLIT_LABEL) -> Tuple[Optional[Iteration], Optional[Dict[str, Any]], bool]:
    """Train (or reuse) and evaluate one configuration on one fold.

    Args:
        it_params (Dict[str, Any]): Iteration hyperparameters.
        fold_label (str): Fold primary key.
        logger (logging.Logger): Logger.
        existing_df (Optional[pd.DataFrame]): Existing results, for resume.
        strategy (str): CV strategy label stored in the row.
        split_label (str): Bundle split label.

    Returns:
        Tuple[Optional[Iteration], Optional[Dict[str, Any]], bool]:
        ``(it, row, skipped)``:

        - fold already complete in ``existing_df`` -> ``(it, None, True)``;
          ``it`` has its data loaded (``ensure_and_load_data`` +
          ``build_windows_normal``) so that dependent calls (baselines in
          :func:`run_cv`) can keep using it without retraining.
        - bundle not found -> ``(None, None, False)``.
        - computed successfully -> ``(it, row, False)``; ``row`` has already
          been saved incrementally to ``RESULTS_CSV``.
    """
    test_year = int(it_params["ano_test"])

    if _is_fold_complete(existing_df, fold_label):
        logger.info("Skipping %s, already present in the CSV", fold_label)
        it = Iteration(it_params)
        it.ensure_and_load_data()
        it.build_windows_normal()
        return it, None, True

    if not _bundle_ready(test_year, split_label=split_label):
        logger.error(
            "[%s] Bundle %s/test%s not found; skipping (generate it first with "
            "dataset_generator.run_expanding_window()/run()).", fold_label, split_label, test_year,
        )
        return None, None, False

    it = Iteration(it_params)
    logger.info(
        "[%s] hash=%s | ano_test=%s | use_pretrain=%s | use_rf=%s | encoder=%s decoder=%s",
        fold_label, it.hash_id, test_year, it.p["use_pretrain"], it.p["use_rf"],
        it.p["encoder_units"], it.p["decoder_units"],
    )

    it.ensure_and_load_data()
    threshold_p90 = _train_threshold_p90(it, fold_label, logger)

    if it.p["use_pretrain"]:
        it.build_windows_qe_input0()
        it.ensure_pretrained()

    it.build_windows_normal()
    preds = it.ensure_predictions(force=False)
    if not preds:
        logger.warning("[%s] No base predictions available; skipping.", fold_label)
        return None, None, False
    it.compute_metrics_for_all_models(force=False)

    row = {
        "fold": fold_label, "test_year": test_year, "hash": it.hash_id,
        "config": it_params.get("ablation_name", "M5"),
        "model": "seq2seq_rf" if it.p["use_rf"] else "seq2seq_base",
        "strategy": strategy,
        "horizon": HORIZON_LABELS.get(int(it.p["offsets"]), f"offset{it.p['offsets']}"),
    }

    if it.p["use_rf"]:
        it.ensure_ensemble_labels(subset="train", force=False)
        it.ensure_ensemble_labels(subset="test", force=False)
        it.train_classifier(force=False)
        it.predict_classifier(force=False)

        algo = str(it.p.get("algorithms", "rf_regressor"))
        cls_npz = os.path.join(it.paths.predictions_cache, f"classifier_{algo}.npz")
        with np.load(cls_npz, allow_pickle=True) as data:
            y_true, y_pred = data["y_true"].astype(float), data["y_pred"].astype(float)
    else:
        with np.load(preds[0]["npz"], allow_pickle=True) as d:
            y_true, y_pred = d["y_true"].astype(float), d["y_pred"].astype(float)

    metrics = log_extended_fold_metrics(
        y_true, y_pred, fold_label=fold_label, threshold_p90=threshold_p90,
        out_dir=str(PREDICTIONS_DIR / "metrics_json"),
    )
    row.update(_flatten_metrics(metrics))
    _save_predictions(fold_label, y_true, y_pred)  # per-fold npz, immediately
    _save_results(row, logger)                      # atomic checkpoint, immediately
    return it, row, False


def _run_heuristic_ensemble_baseline(it: Iteration, year: int, logger: logging.Logger,
                                      existing_df: Optional[pd.DataFrame] = None,
                                      strategy: str = "expanding") -> Optional[Dict[str, Any]]:
    """Evaluate the inverse-MAE weighted ensemble baseline on one fold.

    Args:
        it (Iteration): Iteration whose base models are trained.
        year (int): Test year.
        logger (logging.Logger): Logger.
        existing_df (Optional[pd.DataFrame]): Existing results, for resume.
        strategy (str): CV strategy label.

    Returns:
        Optional[Dict[str, Any]]: The saved row, or ``None`` if skipped or no
        predictions are available.
    """
    # Strategy + horizon suffix in 'fold' (primary key of the CSV, see
    # _save_results) so that expanding/loyo and 24h/48h/72h never collide on
    # the same year -- "expanding" and "48h" add no suffix.
    horizon_tag = HORIZON_LABELS.get(int(it.p["offsets"]), f"offset{it.p['offsets']}")
    fold_suffix = ("" if strategy == "expanding" else f"_{strategy}") + \
                  ("" if horizon_tag == "48h" else f"_{horizon_tag}") + \
                  _variant_suffix()
    fold_label = f"heuristic_ensemble{fold_suffix}_{year}"
    if _is_fold_complete(existing_df, fold_label):
        logger.info("Skipping %s, already present in the CSV", fold_label)
        return None

    # [ANTI-LEAKAGE] The ensemble weights (fit() weights each base model by the
    # inverse of its MAE, see HeuristicEnsembleBaseline.fit in baselines.py) are
    # learned EXCLUSIVELY on train -- fitting them on test and evaluating on
    # that same test would leak the test hit ranking into the metrics.
    preds_train = it.ensure_predictions(force=False, save_csv=False, subset="train")
    preds_test = it.ensure_predictions(force=False, save_csv=False, subset="test")
    if not preds_train or not preds_test:
        return None

    def _stack_preds(preds: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        y = None
        stack = []
        for pr in preds:
            with np.load(pr["npz"], allow_pickle=True) as d:
                if y is None:
                    y = d["y_true"].astype(float)
                stack.append(d["y_pred"].astype(float))
        return y, np.stack(stack, axis=-1)  # (N, n_predictors)

    y_train, X_train = _stack_preds(preds_train)
    y_test, X_test = _stack_preds(preds_test)

    model = HeuristicEnsembleBaseline(mode="weighted")
    model.fit(X_train, y_train)
    logger.info("[heuristic_ensemble] Fitted on Train, evaluating on Test (fold=%s)", fold_label)
    y_pred = model.predict(X_test)[:, 0]

    threshold_p90 = _train_threshold_p90(it, fold_label, logger)
    metrics = log_extended_fold_metrics(
        y_test, y_pred, fold_label=fold_label, threshold_p90=threshold_p90,
        out_dir=str(PREDICTIONS_DIR / "metrics_json"),
    )
    row = {"fold": fold_label, "test_year": year, "hash": it.hash_id,
           "config": "heuristic_ensemble", "model": "baseline", "strategy": strategy,
           "horizon": horizon_tag}
    row.update(_flatten_metrics(metrics))
    _save_predictions(fold_label, y_test, y_pred)
    _save_results(row, logger)
    return row


def _m5_it_params(year: int, split: str = SPLIT_LABEL) -> Dict[str, Any]:
    """Iteration parameters of the full M5 configuration for a given year.

    Args:
        year (int): Test year.
        split (str): Bundle split label.

    Returns:
        Dict[str, Any]: Iteration hyperparameters.
    """
    m5 = next(c for c in ABLATION_CONFIGS if c["name"] == "M5")
    it_params = copy.deepcopy(BASE_ITERATION_DEFAULTS)
    it_params.update({"ano_test": year, "ablation_name": "M5", "split": split})
    it_params.update({k: copy.deepcopy(m5[k]) for k in ABLATION_PROPAGATED_KEYS})
    return it_params


def _final_model_it_params(config_name: str) -> Dict[str, Any]:
    """Iteration parameters of a configuration on the continuous holdout bundle.

    Points to the D_dev/D_holdout bundle (``JunioExpanding/test<HOLDOUT_YEAR>``,
    see ``dataset_generator.run_holdout_bundle``). Used by both
    :func:`run_holdout_evaluation` and :func:`run_robustness` so that both
    share the same ``hash_id``: a single training on 100% of D_dev, not one
    per phase.

    Args:
        config_name (str): Configuration name (``M1``-``M5``).

    Returns:
        Dict[str, Any]: Iteration hyperparameters.
    """
    cfg = next(c for c in ABLATION_CONFIGS if c["name"] == config_name)
    it_params = copy.deepcopy(BASE_ITERATION_DEFAULTS)
    it_params.update({"ano_test": HOLDOUT_YEAR, "ablation_name": config_name, "split": SPLIT_LABEL})
    it_params.update({k: copy.deepcopy(cfg[k]) for k in ABLATION_PROPAGATED_KEYS})
    return it_params


def run_holdout_evaluation(best_config_name: str, logger: logging.Logger, resume: bool = True) -> Optional[Dict[str, Any]]:
    """Evaluate the selected model ONCE on the real holdout D_holdout.

    Reuses :func:`_run_single_config` (which trains on 100% of the bundle's
    ``train.csv`` = D_dev and infers on ``test.csv`` = D_holdout) with
    ``strategy="holdout_final"``, distinct from ``expanding``/``loyo``/
    ``robustness_*``.

    For a selected M5, the generic ``use_rf=True`` branch (nested OOF
    meta-training) is used; reusing the LOYO OOF predictions for the
    meta-learner is not implemented.

    Args:
        best_config_name (str): Selected configuration.
        logger (logging.Logger): Logger.
        resume (bool): Skip if the fold is already complete.

    Returns:
        Optional[Dict[str, Any]]: The saved row, or ``None``.
    """
    it_params = _final_model_it_params(best_config_name)
    horizon_tag = _current_horizon_tag()
    horizon_suffix = ("" if horizon_tag == "48h" else f"_{horizon_tag}") + _variant_suffix()
    fold_label = f"{best_config_name}_holdout_final{horizon_suffix}"
    existing_df = _load_existing_results() if resume else None
    _it, row, _skipped = _run_single_config(
        it_params, fold_label=fold_label, logger=logger, existing_df=existing_df,
        strategy="holdout_final", split_label=SPLIT_LABEL,
    )
    return row


def run_holdout_baselines(logger: logging.Logger, resume: bool = True) -> List[Dict[str, Any]]:
    """Evaluate M1 and the baselines on the holdout, mirroring the CV table.

    :func:`run_holdout_evaluation` only covers the selected model; this adds
    the remaining rows on the same D_dev/D_holdout bundle with
    ``strategy="holdout_final"``:

    - M1: :func:`_run_single_config` with ``_final_model_it_params("M1")``
      (trains on 100% of D_dev, infers on D_holdout).
    - Tabular baselines: ``run_baselines_for_iteration`` and
      :func:`_run_heuristic_ensemble_baseline` on an M5 :class:`Iteration`,
      which needs every grid model trained on D_dev (the heuristic ensemble
      stacks their predictions); ``ensure_pretrained``/``finetune`` train
      them if missing.

    Args:
        logger (logging.Logger): Logger.
        resume (bool): Skip folds already complete.

    Returns:
        List[Dict[str, Any]]: Rows computed in this call.
    """
    rows: List[Dict] = []
    existing_df = _load_existing_results() if resume else None
    horizon_tag = _current_horizon_tag()
    horizon_suffix = ("" if horizon_tag == "48h" else f"_{horizon_tag}") + _variant_suffix()

    it_params_m1 = _final_model_it_params("M1")
    _it_m1, row_m1, _skipped = _run_single_config(
        it_params_m1, fold_label=f"M1_holdout_final{horizon_suffix}", logger=logger, existing_df=existing_df,
        strategy="holdout_final", split_label=SPLIT_LABEL,
    )
    if row_m1 is not None:
        rows.append(row_m1)
    if resume:
        existing_df = _load_existing_results()

    it_params_m5 = _final_model_it_params("M5")
    it = Iteration(it_params_m5)
    it.ensure_and_load_data()
    it.build_windows_qe_input0()
    it.ensure_pretrained()
    it.build_windows_normal()
    it.finetune()
    threshold_p90 = it.get_train_threshold()

    b_folds = [f"{bname}_holdout_final_{HOLDOUT_YEAR}{horizon_suffix}" for bname in SKLEARN_BASELINES]
    if resume and all(_is_fold_complete(existing_df, b) for b in b_folds):
        logger.info("Skipping baselines already present in the CSV: %s", ", ".join(b_folds))
    else:
        try:
            baseline_results = run_baselines_for_iteration(it, threshold_p90=threshold_p90)
            for bname, bmetrics in baseline_results.items():
                if resume and _is_fold_complete(existing_df, f"{bname}_holdout_final_{HOLDOUT_YEAR}{horizon_suffix}"):
                    continue  # never rewrite published rows that are already complete
                brow = {
                    "fold": f"{bname}_holdout_final_{HOLDOUT_YEAR}{horizon_suffix}", "test_year": HOLDOUT_YEAR,
                    "hash": it.hash_id, "config": bname, "model": "baseline", "strategy": "holdout_final",
                    "horizon": horizon_tag,
                }
                brow.update(_flatten_metrics(bmetrics))
                rows.append(brow)
                _save_results(brow, logger)
        except Exception:
            logger.exception("[run_holdout_baselines] Regression baselines failed")

    try:
        row_h = _run_heuristic_ensemble_baseline(
            it, HOLDOUT_YEAR, logger, existing_df=existing_df if resume else None, strategy="holdout_final",
        )
        if row_h is not None:
            rows.append(row_h)
    except Exception:
        logger.exception("[run_holdout_baselines] heuristic_ensemble failed")

    return rows


def run_cv(years: List[int], logger: logging.Logger, resume: bool = True, strategy: str = "expanding",
           split_label: str = SPLIT_LABEL) -> List[Dict[str, Any]]:
    """Run M5 and every baseline on the given CV folds.

    Args:
        years (List[int]): Test years.
        logger (logging.Logger): Logger.
        resume (bool): Skip folds already complete.
        strategy (str): ``"expanding"`` or ``"loyo"``.
        split_label (str): Bundle split label.

    Returns:
        List[Dict[str, Any]]: Rows computed in this call.
    """
    rows: List[Dict] = []
    existing_df = _load_existing_results() if resume else None
    horizon_tag = _current_horizon_tag()
    # Strategy + horizon suffix in 'fold' (primary key of the CSV) so that
    # expanding/loyo and 24h/48h/72h never collide on the same year --
    # "expanding" and "48h" add no suffix, to keep the existing rows of
    # results_consolidated.csv valid.
    fold_suffix = ("" if strategy == "expanding" else f"_{strategy}") + \
                  ("" if horizon_tag == "48h" else f"_{horizon_tag}") + \
                  _variant_suffix()

    for year in years:
        it_params = _m5_it_params(year, split=split_label)

        try:
            it, row, _skipped = _run_single_config(
                it_params, fold_label=f"M5{fold_suffix}_{year}", logger=logger, existing_df=existing_df,
                strategy=strategy, split_label=split_label,
            )
        except Exception:
            logger.exception("[run_cv] M5 failed in %s", year)
            continue
        if it is None:
            continue
        if row is not None:
            rows.append(row)

        # Resume is checked over ALL sklearn baselines, not only the two
        # historical ones: otherwise, when the tau=0.90 variants were added, the
        # tau=0.50 rows already present would skip the fold and the new ones
        # would never be computed.
        b_folds = [f"{bname}{fold_suffix}_{year}" for bname in SKLEARN_BASELINES]
        if resume and all(_is_fold_complete(existing_df, b) for b in b_folds):
            logger.info("Skipping baselines already present in the CSV: %s", ", ".join(b_folds))
        else:
            try:
                threshold_p90 = _train_threshold_p90(it, f"baselines{fold_suffix}_{year}", logger)
                baseline_results = run_baselines_for_iteration(it, threshold_p90=threshold_p90)
                for bname, bmetrics in baseline_results.items():
                    # Only missing rows are saved: a complete published row is
                    # never rewritten, even if recomputed to add another variant.
                    if resume and _is_fold_complete(existing_df, f"{bname}{fold_suffix}_{year}"):
                        continue
                    brow = {"fold": f"{bname}{fold_suffix}_{year}", "test_year": year, "hash": it.hash_id,
                            "config": bname, "model": "baseline", "strategy": strategy,
                            "horizon": horizon_tag}
                    brow.update(_flatten_metrics(bmetrics))
                    rows.append(brow)
                    _save_results(brow, logger)
            except Exception:
                logger.exception("[run_cv] Regression baselines failed in %s", year)

        try:
            row_h = _run_heuristic_ensemble_baseline(
                it, year, logger, existing_df=existing_df if resume else None, strategy=strategy,
            )
            if row_h:
                rows.append(row_h)
        except Exception:
            logger.exception("[run_cv] HeuristicEnsembleBaseline failed in %s", year)

        if resume:
            existing_df = _load_existing_results()  # refresh: see what this year just saved

    return rows


def run_ablation(years: List[int], logger: logging.Logger, resume: bool = True, strategy: str = "expanding",
                  split_label: str = SPLIT_LABEL, configs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Run the ablation configurations M1-M4 on the given CV folds.

    Args:
        years (List[int]): Test years.
        logger (logging.Logger): Logger.
        resume (bool): Skip folds already complete.
        strategy (str): ``"expanding"`` or ``"loyo"``.
        split_label (str): Bundle split label.
        configs (Optional[List[str]]): Restrict to these configuration names.

    Returns:
        List[Dict[str, Any]]: Rows computed in this call.
    """
    rows: List[Dict] = []
    existing_df = _load_existing_results() if resume else None
    horizon_tag = _current_horizon_tag()
    fold_suffix = ("" if strategy == "expanding" else f"_{strategy}") + \
                  ("" if horizon_tag == "48h" else f"_{horizon_tag}") + \
                  _variant_suffix()
    for cfg in ABLATION_CONFIGS:
        if cfg["name"] == "M5":
            continue  # covered by --run_cv
        if configs is not None and cfg["name"] not in configs:
            continue  # --configs: e.g. seeds 42/123 only for M1/M2, without retraining M3/M4
        for year in years:
            it_params = copy.deepcopy(BASE_ITERATION_DEFAULTS)
            it_params.update({"ano_test": year, "ablation_name": cfg["name"], "split": split_label})
            it_params.update({k: copy.deepcopy(cfg[k]) for k in ABLATION_PROPAGATED_KEYS})
            try:
                _it, row, _skipped = _run_single_config(
                    it_params, fold_label=f"{cfg['name']}{fold_suffix}_{year}", logger=logger,
                    existing_df=existing_df, strategy=strategy, split_label=split_label,
                )
            except Exception:
                logger.exception("[run_ablation] %s failed in %s", cfg["name"], year)
                continue
            if row is not None:
                rows.append(row)
            if resume:
                existing_df = _load_existing_results()
    return rows


# --------------------------------------------------------------------------- #
# --recalculate_metrics: recompute M5 + baseline metrics from the saved .npz
# predictions with the correct TRAINING threshold, without retraining.
# --------------------------------------------------------------------------- #

RECALC_CONFIGS = ("M5", "quantile_regression", "gradient_boosting_quantile", "heuristic_ensemble")
# The sklearn baselines never save a .npz (run_baselines_for_iteration in
# main_pipeline.py only returns metrics, not raw predictions). For them, instead
# of "no retraining", a light sklearn refit (seconds, no TensorFlow or GPU) is
# run on the windows already cached on disk by build_windows_normal().
_SKLEARN_REFIT_CONFIGS = tuple(SKLEARN_BASELINES)


def recalculate_metrics_from_saved_predictions(logger: logging.Logger,
                                               configs: Tuple[str, ...] = RECALC_CONFIGS) -> List[Dict[str, Any]]:
    """Recompute the metrics of existing rows with the training threshold.

    Iterates over the rows of ``results_consolidated.csv`` for ``configs``
    (M5 and the baselines by default), recomputes NSE/KGE/HitRatio/FARate/
    FARatio/F1/Precision/Recall/peak timing/bootstrap CIs with the threshold
    fixed on TRAIN (not on test) and upserts the CSV atomically.

    M5 and ``heuristic_ensemble`` reread the saved ``predictions/<fold>.npz``
    (no retraining, instantaneous). The sklearn baselines have no persisted
    ``.npz`` (see ``_SKLEARN_REFIT_CONFIGS``) and are refitted on the cached
    windows (seconds; the Seq2Seq models and the RF are not involved).

    Args:
        logger (logging.Logger): Logger.
        configs (Tuple[str, ...]): Configurations to recompute.

    Returns:
        List[Dict[str, Any]]: Updated rows.
    """
    existing_df = _load_existing_results()
    if existing_df is None:
        logger.warning("[recalculate_metrics] %s does not exist or is empty; nothing to recompute.", RESULTS_CSV)
        return []

    rows_to_fix = existing_df[existing_df["config"].isin(configs)]
    # Only rows of the PUBLISHED variant are re-evaluated: they are the ones
    # with consolidated .npz files. Rows of other variants (p95, another seed)
    # are excluded, to avoid recomputing on top of a recomputation or chaining
    # suffixes.
    if "threshold_percentile" in rows_to_fix.columns:
        base_pct = rows_to_fix["threshold_percentile"].fillna(PUBLISHED_THRESHOLD_PCT)
        rows_to_fix = rows_to_fix[base_pct == PUBLISHED_THRESHOLD_PCT]
    if "seed" in rows_to_fix.columns:
        base_seed = rows_to_fix["seed"].fillna(PUBLISHED_SEED)
        rows_to_fix = rows_to_fix[base_seed == _current_seed()]
    if "horizon" in rows_to_fix.columns:
        rows_to_fix = rows_to_fix[rows_to_fix["horizon"] == _current_horizon_tag()]
    # The threshold is rebuilt with _m5_it_params(year), which points to the
    # EXPANDING split (SPLIT_LABEL). That is correct for expanding rows and for
    # the holdout (JunioExpanding/test2025; verified identical to the real run
    # at p90 and p95), but NOT for LOYO, whose training set lives in
    # LOYO_SPLIT_LABEL and has a different percentile -- and for 2014-2016 no
    # expanding bundle even exists, so a spurious one would be generated. LOYO
    # is therefore excluded.
    if "strategy" in rows_to_fix.columns:
        rows_to_fix = rows_to_fix[rows_to_fix["strategy"].isin(("expanding", "holdout_final"))]
    if rows_to_fix.empty:
        logger.warning("[recalculate_metrics] No rows for configs=%s.", configs)
        return []

    # With the published variant the suffix is "" and the same row is
    # rewritten (historical behavior); with p95 a NEW row is emitted from the
    # SAME predictions, re-evaluated against the other training threshold.
    variant_suffix = _variant_suffix()

    it_cache: Dict[int, Iteration] = {}
    threshold_cache: Dict[int, float] = {}
    sklearn_refit_cache: Dict[int, Dict[str, Dict]] = {}
    updated_rows: List[Dict] = []

    def _get_iteration(year: int, fold_label: str) -> Iteration:
        if year not in it_cache:
            it = Iteration(_m5_it_params(year))
            it.ensure_and_load_data()
            it_cache[year] = it
            threshold_cache[year] = _train_threshold_p90(it, fold_label, logger)
        return it_cache[year]

    for _, existing_row in rows_to_fix.iterrows():
        fold_label = str(existing_row["fold"])
        year = int(existing_row["test_year"])
        config = str(existing_row["config"])
        npz_path = PREDICTIONS_DIR / f"{fold_label}.npz"

        if npz_path.exists():
            with np.load(npz_path, allow_pickle=True) as d:
                y_true, y_pred = d["y_true"].astype(float), d["y_pred"].astype(float)
            _get_iteration(year, fold_label)  # only to populate threshold_cache[year]
            metrics = log_extended_fold_metrics(
                y_true, y_pred, fold_label=fold_label,
                threshold_p90=threshold_cache[year],
                out_dir=str(PREDICTIONS_DIR / "metrics_json"),
            )
        elif config in _SKLEARN_REFIT_CONFIGS:
            it = _get_iteration(year, fold_label)
            if year not in sklearn_refit_cache:
                sklearn_refit_cache[year] = run_baselines_for_iteration(
                    it, threshold_p90=threshold_cache[year],
                )
            # run_baselines_for_iteration already computed the metrics with the
            # correct threshold; there are no raw y_true/y_pred to re-evaluate.
            metrics = sklearn_refit_cache[year][config]
        else:
            logger.warning("[recalculate_metrics] %s: %s does not exist; skipped.", fold_label, npz_path)
            continue

        row = existing_row.to_dict()
        row.update(_flatten_metrics(metrics))
        row["fold"] = _apply_variant_suffix(fold_label, variant_suffix)
        updated_rows.append(row)
        logger.info("[recalculate_metrics] %s -> %s: training threshold = %.6f",
                    fold_label, row["fold"], threshold_cache[year])

    _save_results(updated_rows, logger)
    logger.info("[recalculate_metrics] %d rows recomputed.", len(updated_rows))
    return updated_rows


# --------------------------------------------------------------------------- #
# --run_robustness (Monte Carlo rainfall perturbation)
# --------------------------------------------------------------------------- #

def _build_predict_fn(it: Iteration, algo_tag: str) -> Callable[[pd.DataFrame], np.ndarray]:
    """Build a single-branch inference function for raw (possibly perturbed) data.

    The returned ``predict_fn(df_raw)`` yields the denormalized prediction of
    the already fine-tuned base model ``algo_tag``. Windows are rebuilt with
    ``prepare_model_inputs`` (the same code used by
    ``Iteration._build_windows_from_dfs``, see tests/test_leakage.py) and
    normalized with the TRAINING normalization parameters, never recomputed on
    the perturbed data.

    Args:
        it (Iteration): Iteration with data loaded and models fine-tuned.
        algo_tag (str): Model tag, e.g. ``pinball_from_penalty_2``.

    Returns:
        Callable[[pd.DataFrame], np.ndarray]: The inference function.
    """
    import tensorflow as tf

    manifest = it.get_manifest()
    columns_order = list(manifest["columns_order"])
    variable_salida = manifest.get("target_col", "Qe")
    historicos = [c for c in columns_order if not c.startswith("pred")]
    predicciones = [c for c in columns_order if c.startswith("pred")]
    contextos = int(it.p["contextos"])
    offset = int(it.p["offsets"])
    norm_params = it._norm_params
    # preds_seq has offset+1 steps [t+1...t+offset+1]; the evaluated horizon is
    # the LAST one (index=offset), not offset-1.
    step_idx = offset

    enc_path = os.path.join(it.paths.models, f"encoder_{algo_tag}.keras")
    dec_path = os.path.join(it.paths.models, f"decoder_{algo_tag}.keras")
    proj_path = os.path.join(it.paths.models, f"projector_{algo_tag}.keras")
    encoder_model = tf.keras.models.load_model(enc_path)
    decoder_model = tf.keras.models.load_model(dec_path)
    projector_model = tf.keras.models.load_model(proj_path) if os.path.exists(proj_path) else None

    def predict_fn(df_raw: pd.DataFrame) -> np.ndarray:
        df_norm = normalize_with_params(df_raw.copy(), norm_params)
        _, _, (x_enc, x_dec, _y_norm, _dates) = prepare_model_inputs(
            train_norm=df_norm, val_norm=None, test_norm=df_norm,
            variable_salida=variable_salida, historicos=historicos, predicciones=predicciones,
            contextos=contextos, offset=offset,
        )
        preds_seq = it._predict_iterative(encoder_model, decoder_model, x_enc, x_dec, projector_model=projector_model)
        y_pred_norm = preds_seq[:, step_idx, 0]
        return it._denorm_qe(y_pred_norm)

    return predict_fn


def _build_predict_fn_m5(it: Iteration, algo_tags: List[str], clf: Any,
                         meta: Dict[str, Any]) -> Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:
    """Build the full M5 inference function (all Seq2Seq branches + RF selector).

    The returned ``predict_fn(df_raw) -> (y_pred, cls_idx)`` reproduces EXACTLY
    the M5 inference pipeline, unlike :func:`_build_predict_fn` (a single
    branch, M1-M4). It mirrors :meth:`Iteration.predict_classifier` on fresh
    (possibly perturbed) data instead of the cached test ``.npz`` files: same
    training normalization, same branch order (``meta["model_bases"]``, the
    order the RF was trained with, not the order of the penalty grid), same
    rounding/clipping of ``predicted_continuous_idx -> cls_idx -> y_cls``.

    Pure inference: only ``tf.keras.models.load_model`` on saved ``.keras``
    files, never training. The caller has already checked that every file
    exists.

    Args:
        it (Iteration): Iteration with data loaded.
        algo_tags (List[str]): Branch tags.
        clf (Any): Trained meta-learner.
        meta (Dict[str, Any]): Meta-learner metadata (``model_bases``).

    Returns:
        Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray]]: The inference
        function.
    """
    import tensorflow as tf

    manifest = it.get_manifest()
    columns_order = list(manifest["columns_order"])
    variable_salida = manifest.get("target_col", "Qe")
    historicos = [c for c in columns_order if not c.startswith("pred")]
    predicciones = [c for c in columns_order if c.startswith("pred")]
    contextos = int(it.p["contextos"])
    offset = int(it.p["offsets"])
    norm_params = it._norm_params
    step_idx = offset

    branches_by_tag = {}
    for tag in algo_tags:
        enc = tf.keras.models.load_model(os.path.join(it.paths.models, f"encoder_{tag}.keras"))
        dec = tf.keras.models.load_model(os.path.join(it.paths.models, f"decoder_{tag}.keras"))
        proj_path = os.path.join(it.paths.models, f"projector_{tag}.keras")
        proj = tf.keras.models.load_model(proj_path) if os.path.exists(proj_path) else None
        branches_by_tag[tag] = (enc, dec, proj)

    bases_want = [_canonical_base(b) for b in meta["model_bases"]]
    ordered_branches = [(b,) + branches_by_tag[b] for b in bases_want]

    def predict_fn(df_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df_norm = normalize_with_params(df_raw.copy(), norm_params)
        _, _, (x_enc, x_dec, _y_norm, _dates) = prepare_model_inputs(
            train_norm=df_norm, val_norm=None, test_norm=df_norm,
            variable_salida=variable_salida, historicos=historicos, predicciones=predicciones,
            contextos=contextos, offset=offset,
        )
        preds_stack = []
        for _base, enc, dec, proj in ordered_branches:
            preds_seq = it._predict_iterative(enc, dec, x_enc, x_dec, projector_model=proj)
            y_pred_norm = preds_seq[:, step_idx, 0]
            preds_stack.append(it._denorm_qe(y_pred_norm))
        predictions_all = np.vstack(preds_stack)  # (num_models, N)
        base_preds = predictions_all.T            # (N, num_models)

        predicted_continuous_idx = clf.predict([x_enc, x_dec, base_preds])
        cls_idx = np.clip(np.round(predicted_continuous_idx).astype(int), 0, len(ordered_branches) - 1)
        y_cls = predictions_all[cls_idx, np.arange(predictions_all.shape[1])]
        return y_cls, cls_idx

    return predict_fn


def run_robustness(years: List[int], logger: logging.Logger, best_config_name: str, n_replicas: int = 50,
                    resume: bool = True) -> List[Dict[str, Any]]:
    """Monte Carlo rainfall-perturbation test on a single-branch selected model.

    The inference function depends on the selected configuration. For
    M1-M4 (a single Seq2Seq model, no RF selector) the iteration parameters
    and model tag come from the real configuration, sharing the ``hash_id`` of
    :func:`run_holdout_evaluation` through :func:`_final_model_it_params` (one
    training on 100% of D_dev). Configurations with the RF selector are
    handled by :func:`run_robustness_m5`; here they are logged and skipped
    instead of producing an incorrect result.

    Args:
        years (List[int]): Test years (normally the holdout year).
        logger (logging.Logger): Logger.
        best_config_name (str): Selected configuration.
        n_replicas (int): Monte Carlo replicas per noise type.
        resume (bool): Skip folds already complete.

    Returns:
        List[Dict[str, Any]]: Rows computed in this call.
    """
    rows: List[Dict] = []
    cfg = next(c for c in ABLATION_CONFIGS if c["name"] == best_config_name)
    if cfg["use_rf"]:
        logger.error(
            "[run_robustness] %s uses the RF selector -- the inference function for "
            "that branch is not implemented here (requires branch_idx + spurious "
            "branch-switch rate). Nothing to do.",
            best_config_name,
        )
        return rows

    algo_tag = f"{cfg['loss_name']}_{int(cfg['penalty'][0])}"
    existing_df = _load_existing_results() if resume else None
    # Horizon suffix in the fold and .npz names so that 24h/72h never overwrite
    # the robustness_{config}_{noise}_{year}.npz files already generated for
    # 48h -- "48h" adds no suffix (same criterion as run_cv/run_ablation). The
    # resulting 5-part name (config_noise_year_horizon) is silently ignored by
    # _robustness_file_groups in generate_paper_figures.py, which only handles
    # the 4-part pattern, so the existing robustness figures stay intact.
    horizon_tag = _current_horizon_tag()
    horizon_suffix = ("" if horizon_tag == "48h" else f"_{horizon_tag}") + _variant_suffix()

    for year in years:
        u_fold = f"robustness_{best_config_name}_uniform_{year}{horizon_suffix}"
        g_fold = f"robustness_{best_config_name}_gaussian_{year}{horizon_suffix}"
        if resume and _is_fold_complete(existing_df, u_fold, ROBUSTNESS_COLUMNS) \
                and _is_fold_complete(existing_df, g_fold, ROBUSTNESS_COLUMNS):
            logger.info("Skipping %s and %s, already present in the CSV", u_fold, g_fold)
            continue

        if not _bundle_ready(year, split_label=SPLIT_LABEL):
            logger.error("[run_robustness] Bundle not ready for %s; skipping.", year)
            continue

        it_params = _final_model_it_params(best_config_name)
        it_params["ano_test"] = year  # normally HOLDOUT_YEAR -- shares hash/weights with the holdout evaluation

        try:
            it = Iteration(it_params)
            it.ensure_and_load_data()
            if it.p["use_pretrain"]:
                it.build_windows_qe_input0()
                it.ensure_pretrained()
            it.build_windows_normal()
            it.finetune()

            predict_fn = _build_predict_fn(it, algo_tag)

            bundle_base = dataio.resolve_bundle_base(CONFIG, SPLIT_LABEL, year)
            raw_test_df = pd.read_csv(os.path.join(bundle_base, "test.csv"))

            # y_true/threshold_p90 aligned with the same windows produced by
            # predict_fn(raw_test_df) (same bundle, same prepare_model_inputs
            # call), so that delta HitRatio/FARate can be computed, not only the
            # raw spread of Qe across replicas.
            _, _, y_te, _ = it.test_inputs
            y_true_holdout = it._denorm_qe(y_te[:, -1].astype(float))
            threshold_p90 = it.get_train_threshold()

            result = run_perturbation_test(
                predict_fn, raw_test_df, n_replicas=n_replicas,
                uniform_frac=CONFIG.robustness["uniform_frac"],
                gaussian_frac=CONFIG.robustness["gaussian_frac"],
                seed=CONFIG.robustness["seed"],
                y_true=y_true_holdout, threshold_p90=threshold_p90,
            )
        except Exception:
            logger.exception("[run_robustness] Failed in %s", year)
            continue

        for noise_type in ("uniform", "gaussian"):
            r = result[noise_type]
            PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                PREDICTIONS_DIR / f"robustness_{best_config_name}_{noise_type}_{year}{horizon_suffix}.npz",
                predictions=r["predictions"], mean=r["mean"], std=r["std"], baseline=r["baseline"],
            )
            row = {
                "fold": f"robustness_{best_config_name}_{noise_type}_{year}{horizon_suffix}", "test_year": year,
                "config": f"{best_config_name}_robustness_{noise_type}", "model": "seq2seq_base",
                "horizon": horizon_tag,
                "n_replicas": r["n_replicas"], "frac": r["frac"],
                "mean_abs_deviation_from_baseline": float(np.mean(np.abs(r["mean"] - r["baseline"]))),
                "mean_std_across_replicas": float(np.mean(r["std"])),
                "max_std_across_replicas": float(np.max(r["std"])),
                "delta_hit_ratio": r.get("delta_hit_ratio"),
                "delta_farate": r.get("delta_farate"),
            }
            rows.append(row)
            _save_results(row, logger)  # atomic checkpoint, immediately per noise type
            logger.info(
                "[run_robustness] %s/%s/%s: mean deviation |mean-baseline|=%.4f, mean std across replicas=%.4f",
                best_config_name, year, noise_type,
                row["mean_abs_deviation_from_baseline"], row["mean_std_across_replicas"],
            )

        if resume:
            existing_df = _load_existing_results()

    return rows


def run_robustness_m5(years: List[int], logger: logging.Logger, n_replicas: int = 50,
                      resume: bool = True) -> List[Dict[str, Any]]:
    """Monte Carlo rainfall-perturbation test on the full M5 framework.

    Reproduces the real M5 inference pipeline (every Seq2Seq branch + RF
    selector) through :func:`_build_predict_fn_m5`, instead of the
    single-branch :func:`_build_predict_fn`. Adds ``rf_branch_stability``: the
    fraction of (replica, time step) pairs in which the RF selects the SAME
    branch as on the unperturbed input.

    Pure inference: only loads already trained/cached weights and never calls
    ``ensure_pretrained``/``finetune``/``train_classifier``. If any file is
    missing, that year fails explicitly instead of retraining.

    Args:
        years (List[int]): Test years.
        logger (logging.Logger): Logger.
        n_replicas (int): Monte Carlo replicas per noise type.
        resume (bool): Skip folds already complete.

    Returns:
        List[Dict[str, Any]]: Rows computed in this call.
    """
    rows: List[Dict] = []
    cfg = next(c for c in ABLATION_CONFIGS if c["name"] == "M5")
    trained_penalties = [p for p in cfg["penalty"] if p != 0] or cfg["penalty"]
    algo_tags = [f"{cfg['loss_name']}_{int(p)}" for p in trained_penalties]
    existing_df = _load_existing_results() if resume else None
    required_cols = ROBUSTNESS_COLUMNS + ["rf_branch_stability"]
    horizon_tag = _current_horizon_tag()
    horizon_suffix = ("" if horizon_tag == "48h" else f"_{horizon_tag}") + _variant_suffix()

    for year in years:
        u_fold = f"robustness_M5_uniform_{year}{horizon_suffix}"
        g_fold = f"robustness_M5_gaussian_{year}{horizon_suffix}"
        if resume and _is_fold_complete(existing_df, u_fold, required_cols) \
                and _is_fold_complete(existing_df, g_fold, required_cols):
            logger.info("Skipping %s and %s, already present in the CSV", u_fold, g_fold)
            continue

        if not _bundle_ready(year, split_label=SPLIT_LABEL):
            logger.error("[run_robustness_m5] Bundle not ready for %s; skipping.", year)
            continue

        it_params = _final_model_it_params("M5")
        it_params["ano_test"] = year

        try:
            it = Iteration(it_params)
            it.ensure_and_load_data()
            it.build_windows_normal()

            # Pure inference: check the cache before touching anything -- if
            # any weight file is missing, fail instead of retraining.
            missing = [
                p for tag in algo_tags for p in (
                    os.path.join(it.paths.models, f"encoder_{tag}.keras"),
                    os.path.join(it.paths.models, f"decoder_{tag}.keras"),
                )
                if not os.path.exists(p)
            ]
            algo = str(it.p.get("algorithms", "rf_regressor"))
            clf, meta = it._load_classifier(algo)
            if clf is None:
                missing.append(f"classifier {algo}")
            if missing:
                logger.error(
                    "[run_robustness_m5] %s: trained weights missing (%s); "
                    "not retraining, skipping.", year, missing,
                )
                continue

            predict_fn = _build_predict_fn_m5(it, algo_tags, clf, meta)

            bundle_base = dataio.resolve_bundle_base(CONFIG, SPLIT_LABEL, year)
            raw_test_df = pd.read_csv(os.path.join(bundle_base, "test.csv"))

            _, _, y_te, _ = it.test_inputs
            y_true_holdout = it._denorm_qe(y_te[:, -1].astype(float))
            threshold_p90 = it.get_train_threshold()

            uniform_frac = CONFIG.robustness["uniform_frac"]
            gaussian_frac = CONFIG.robustness["gaussian_frac"]
            seed = int(CONFIG.robustness["seed"])

            rng = np.random.default_rng(seed)
            baseline_pred, baseline_cls_idx = predict_fn(raw_test_df)
            baseline_extreme = compute_extreme_metrics(y_true_holdout, baseline_pred, threshold_p90)

            result = {}
            for noise_type, perturb_fn, frac in (
                ("uniform", perturb_rainfall_uniform, uniform_frac),
                ("gaussian", perturb_rainfall_gaussian, gaussian_frac),
            ):
                replicas_pred, replicas_cls = [], []
                for _ in range(n_replicas):
                    rep_seed = int(rng.integers(0, 2**32 - 1))
                    df_pert = perturb_fn(raw_test_df, seed=rep_seed, **(
                        {"frac": frac} if noise_type == "uniform" else {"sigma_frac": frac}
                    ))
                    y_pred, cls_idx = predict_fn(df_pert)
                    replicas_pred.append(y_pred)
                    replicas_cls.append(cls_idx)

                stacked = np.stack(replicas_pred, axis=0)
                mean_pred = stacked.mean(axis=0)
                cls_stack = np.stack(replicas_cls, axis=0)
                perturbed_extreme = compute_extreme_metrics(y_true_holdout, mean_pred, threshold_p90)
                result[noise_type] = {
                    "predictions": stacked, "mean": mean_pred, "std": stacked.std(axis=0),
                    "baseline": baseline_pred, "n_replicas": n_replicas, "frac": frac,
                    "delta_hit_ratio": perturbed_extreme["HitRatio"] - baseline_extreme["HitRatio"],
                    "delta_farate": perturbed_extreme["FARate"] - baseline_extreme["FARate"],
                    "rf_branch_stability": float(np.mean(cls_stack == baseline_cls_idx[None, :])),
                }
        except Exception:
            logger.exception("[run_robustness_m5] Failed in %s", year)
            continue

        for noise_type in ("uniform", "gaussian"):
            r = result[noise_type]
            PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                PREDICTIONS_DIR / f"robustness_M5_{noise_type}_{year}{horizon_suffix}.npz",
                predictions=r["predictions"], mean=r["mean"], std=r["std"], baseline=r["baseline"],
            )
            row = {
                "fold": f"robustness_M5_{noise_type}_{year}{horizon_suffix}", "test_year": year,
                "config": f"M5_robustness_{noise_type}", "model": "seq2seq_rf", "horizon": horizon_tag,
                "n_replicas": r["n_replicas"], "frac": r["frac"],
                "mean_abs_deviation_from_baseline": float(np.mean(np.abs(r["mean"] - r["baseline"]))),
                "mean_std_across_replicas": float(np.mean(r["std"])),
                "max_std_across_replicas": float(np.max(r["std"])),
                "delta_hit_ratio": r["delta_hit_ratio"],
                "delta_farate": r["delta_farate"],
                "rf_branch_stability": r["rf_branch_stability"],
            }
            rows.append(row)
            _save_results(row, logger)
            logger.info(
                "[run_robustness_m5] M5/%s/%s: MAD=%.4f sigma_mean=%.4f sigma_max=%.4f "
                "dHitRatio=%.4f dFARate=%.4f branch_stability=%.4f",
                year, noise_type, row["mean_abs_deviation_from_baseline"],
                row["mean_std_across_replicas"], row["max_std_across_replicas"],
                row["delta_hit_ratio"], row["delta_farate"], row["rf_branch_stability"],
            )

        if resume:
            existing_df = _load_existing_results()

    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    """Parse the command line and dispatch the requested experiment phases."""
    parser = argparse.ArgumentParser(description="Phase 3 - Flowpredictor experiment orchestrator")
    parser.add_argument("--run_cv", action="store_true", help="M5 + baselines on the expanding-window CV (test years 2017-2023)")
    parser.add_argument("--run_ablation", action="store_true", help="M1-M4 on the expanding-window CV")
    parser.add_argument("--run_cv_loyo", action="store_true",
                         help="M5 + baselines on the LOYO CV ('Junio' bundles, truncated before the holdout)")
    parser.add_argument("--run_ablation_loyo", action="store_true", help="M1-M4 on the LOYO CV")
    parser.add_argument("--run_holdout_final", action="store_true",
                         help="Evaluate the already selected model M* (selected on the existing "
                              "results_consolidated.csv) ONCE on the real holdout D_holdout.")
    parser.add_argument("--run_robustness", action="store_true",
                         help="Monte Carlo rainfall perturbation on the already selected model M* "
                              "(configuration-dependent inference function).")
    parser.add_argument("--recalculate_metrics", action="store_true",
                         help="Recompute NSE/KGE/HitRatio/F1/... of M5 + baselines from the saved "
                              ".npz predictions with the TRAINING threshold (not test). No retraining.")
    parser.add_argument("--all", action="store_true", help="run_cv + run_ablation + run_robustness")
    parser.add_argument("--years", type=int, nargs="*", default=None,
                         help="Restrict the years of run_cv/run_ablation (default: 2017-2023; "
                              "the 2025 holdout is excluded from the expanding-window CV).")
    parser.add_argument("--loyo_years", type=int, nargs="*", default=None,
                         help=f"Restrict the years of run_cv_loyo/run_ablation_loyo "
                              f"(default: {LOYO_CV_YEARS[0]}-{LOYO_CV_YEARS[-1]}; years without a "
                              f"generated 'Junio' bundle are skipped at run time).")
    parser.add_argument("--robustness_years", type=int, nargs="*", default=[HOLDOUT_YEAR],
                         help="Years for run_robustness (default: only the 2025 holdout).")
    parser.add_argument("--n_replicas", type=int, default=int(CONFIG.robustness["n_replicas"]))
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_epochs_clasificador", type=int, default=None)
    parser.add_argument("--penalty", type=int, nargs="*", default=None,
                         help="Penalty grid (default: [0, 2, 4, 6, 8, 10]).")
    parser.add_argument("--offset", type=int, default=None, choices=[0, 1, 2],
                         help="Evaluated forecast horizon -- 0=24h, 1=48h (default, validated), "
                              "2=72h. Changes the hash_id (Iteration includes 'offsets' in its "
                              "hash_payload), so each horizon uses its own model/window cache "
                              "without overwriting another's.")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Training seed (default {PUBLISHED_SEED}, the published run). Another "
                             f"seed trains in its own cache hash and labels its rows with the suffix "
                             f"_s<seed>: it never reuses or overwrites the weights of {PUBLISHED_SEED}.")
    parser.add_argument("--threshold_percentile", type=int, default=None, choices=[90, 95],
                        help=f"Percentile of the extreme-event threshold (default {PUBLISHED_THRESHOLD_PCT}). "
                             "Already precomputed in each bundle's normalization_params.json, excluding "
                             "imputed rows. With --recalculate_metrics it is re-evaluated without retraining.")
    parser.add_argument("--configs", type=str, nargs="*", default=None,
                        help="Restrict --run_ablation to these configurations (e.g. M1 M2). Default: M1-M4.")
    parser.add_argument("--recalc_configs", type=str, nargs="*", default=None,
                        help="Configurations to re-evaluate with --recalculate_metrics (default: M5 + baselines).")
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                         help="Epochs without train_loss improvement before stopping (default: 10; "
                              "pass 0 or a negative value to disable it and always run max_epochs).")
    parser.add_argument("--no_resume", action="store_true",
                         help="Ignore results_consolidated.csv and recompute every fold even if "
                              "already present (default: automatic resume enabled).")
    args = parser.parse_args()
    resume = not args.no_resume

    if args.max_epochs is not None:
        BASE_ITERATION_DEFAULTS["max_epochs"] = args.max_epochs
    if args.max_epochs_clasificador is not None:
        BASE_ITERATION_DEFAULTS["max_epochs_clasificador"] = args.max_epochs_clasificador
    if args.penalty is not None:
        BASE_ITERATION_DEFAULTS["penalty"] = args.penalty
    if args.offset is not None:
        BASE_ITERATION_DEFAULTS["offsets"] = args.offset
    if args.seed is not None:
        BASE_ITERATION_DEFAULTS["seed"] = args.seed
    if args.threshold_percentile is not None:
        # eval_threshold is the key Iteration.get_train_threshold() looks up in
        # norm_params["Qe"], and it enters the hash: p95 never shares a cache with p90.
        BASE_ITERATION_DEFAULTS["eval_threshold"] = f"p{args.threshold_percentile}"
    if args.early_stopping_patience is not None:
        BASE_ITERATION_DEFAULTS["early_stopping_patience"] = (
            args.early_stopping_patience if args.early_stopping_patience > 0 else None
        )

    if not (args.run_cv or args.run_ablation or args.run_cv_loyo or args.run_ablation_loyo
            or args.run_holdout_final or args.run_robustness or args.recalculate_metrics or args.all):
        parser.error(
            "Specify at least one of --run_cv/--run_ablation/--run_cv_loyo/"
            "--run_ablation_loyo/--run_holdout_final/--run_robustness/--recalculate_metrics/--all"
        )

    init_logger(logging.INFO, numero_prueba=NUMERO_PRUEBA)
    logger = get_logger()

    if args.recalculate_metrics:
        logger.info("=== --recalculate_metrics: M5 + baselines from saved .npz files (no retraining) ===")
        recalculate_metrics_from_saved_predictions(
            logger, configs=tuple(args.recalc_configs) if args.recalc_configs else RECALC_CONFIGS,
        )
        if not (args.run_cv or args.run_ablation or args.run_cv_loyo or args.run_ablation_loyo
                or args.run_holdout_final or args.run_robustness or args.all):
            return

    plan = build_expanding_window_cv_plan()
    # The is_holdout=True fold (HOLDOUT_YEAR) is excluded here: run_cv/run_ablation
    # are the expanding-window CV, not the final holdout evaluation.
    years = args.years if args.years else [f["test_year"] for f in plan if not f["is_holdout"]]
    loyo_years = args.loyo_years if args.loyo_years else LOYO_CV_YEARS

    logger.info("=== PHASE 3: run_experiments_phase3.py ===")
    logger.info(
        "years=%s | loyo_years=%s | robustness_years=%s | offset=%s (horizon=%s) | max_epochs=%s | "
        "max_epochs_clasificador=%s | penalty=%s | early_stopping_patience=%s | resume=%s",
        years, loyo_years, args.robustness_years, BASE_ITERATION_DEFAULTS["offsets"], _current_horizon_tag(),
        BASE_ITERATION_DEFAULTS["max_epochs"],
        BASE_ITERATION_DEFAULTS["max_epochs_clasificador"], BASE_ITERATION_DEFAULTS["penalty"],
        BASE_ITERATION_DEFAULTS["early_stopping_patience"], resume,
    )

    # Incremental saving: each fold (and each baseline/noise type within a
    # fold) is written to RESULTS_CSV as soon as it is computed (see
    # _save_results), not at the end of each phase, so an interruption in the
    # middle of --all keeps everything already computed. all_rows is only used
    # for the final summary in the log.
    t0 = time.time()
    all_rows: List[Dict] = []

    if args.run_cv or args.all:
        logger.info("=== --run_cv: M5 + baselines on %d years (expanding) ===", len(years))
        all_rows.extend(run_cv(years, logger, resume=resume, strategy="expanding", split_label=SPLIT_LABEL))

    if args.run_ablation or args.all:
        logger.info("=== --run_ablation: M1-M4 on %d years (expanding) ===", len(years))
        all_rows.extend(run_ablation(years, logger, resume=resume, strategy="expanding", split_label=SPLIT_LABEL,
                                     configs=args.configs))

    if args.run_cv_loyo:
        logger.info("=== --run_cv_loyo: M5 + baselines on %d years (loyo) ===", len(loyo_years))
        all_rows.extend(run_cv(loyo_years, logger, resume=resume, strategy="loyo", split_label=LOYO_SPLIT_LABEL))

    if args.run_ablation_loyo:
        logger.info("=== --run_ablation_loyo: M1-M4 on %d years (loyo) ===", len(loyo_years))
        all_rows.extend(run_ablation(loyo_years, logger, resume=resume, strategy="loyo", split_label=LOYO_SPLIT_LABEL,
                                     configs=args.configs))

    if args.run_holdout_final:
        logger.info("=== --run_holdout_final: final evaluation of M* on D_holdout ===")
        try:
            selection = run_model_selection(str(RESULTS_CSV))
            best_config = selection["selected"]
            logger.info("[run_holdout_final] Selected M*: %s (decision_fn=%s)",
                        best_config, selection["decision_fn"])
            row = run_holdout_evaluation(best_config, logger, resume=resume)
            if row is not None:
                all_rows.append(row)
            all_rows.extend(run_holdout_baselines(logger, resume=resume))
            report = build_final_report(str(RESULTS_CSV), out_json_path=str(PREDICTIONS_DIR / "final_report.json"))
            logger.info("[run_holdout_final] Final report saved (M*=%s, holdout_metrics=%s).",
                        report["selected"], "OK" if report["holdout_metrics"] is not None else "pending")
        except Exception:
            logger.exception("[run_holdout_final] Failed to select M* or evaluate it on the holdout")

    if args.run_robustness or args.all:
        try:
            selection = run_model_selection(str(RESULTS_CSV))
            best_config = selection["selected"]
            logger.info("=== --run_robustness: %d MC replicas on %s (M*=%s) ===",
                        args.n_replicas, args.robustness_years, best_config)
            all_rows.extend(run_robustness(
                args.robustness_years, logger, best_config, n_replicas=args.n_replicas, resume=resume,
            ))
        except Exception:
            logger.exception("[run_robustness] Failed to select M* or run the robustness test")

    logger.info("=== PHASE 3 completed in %.1f min (%d new/recomputed rows in this run) ===",
                (time.time() - t0) / 60.0, len(all_rows))


if __name__ == "__main__":
    main()
