"""Integrity tests of the consolidated results and prediction arrays.

Covers resultados/prueba63/results_consolidated.csv (written by
run_experiments_phase3.py) and resultados/prueba63/predictions/*.npz:
  - No NaN/Inf in the metric columns.
  - Completeness: required columns present and non-empty.
  - Consistent confidence intervals (lower <= point <= upper).

The checks are first exercised on synthetic data (clean and deliberately
corrupted), so the tests are meaningful before any real experiment runs; if
results_consolidated.csv / predictions/ exist, the real results are validated
too (skipped otherwise). Also covers incremental checkpointing and resume.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

import resultados.prueba63.src.run_experiments_phase3 as phase3

# Imported from the orchestrator itself (not reimplemented) so that the test's
# "complete fold" criterion never diverges from the one used by the real resume.
METRIC_COLUMNS = phase3.METRIC_COLUMNS
ROBUSTNESS_COLUMNS = phase3.ROBUSTNESS_COLUMNS
REQUIRED_COLUMNS = ["fold", "test_year", "config", "model"]

_PRUEBA_ROOT = Path(__file__).resolve().parents[1] / "resultados" / "prueba63"
RESULTS_CSV = _PRUEBA_ROOT / "results_consolidated.csv"
PREDICTIONS_DIR = _PRUEBA_ROOT / "predictions"


# --------------------------------------------------------------------------- #
# Integrity functions under test
# --------------------------------------------------------------------------- #

def find_nan_inf(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Rows containing NaN or Inf in `columns`."""
    present = [c for c in columns if c in df.columns]
    if not present:
        return pd.DataFrame()
    sub = df[present].apply(pd.to_numeric, errors="coerce")
    bad_mask = sub.isna().any(axis=1) | np.isinf(sub.to_numpy(dtype=float)).any(axis=1)
    return df.loc[bad_mask]


def check_required_columns(df: pd.DataFrame, required: List[str] = REQUIRED_COLUMNS) -> List[str]:
    """Required columns that are missing or entirely empty (all NaN)."""
    missing = [c for c in required if c not in df.columns]
    empty = [c for c in required if c in df.columns and df[c].isna().all()]
    return missing + empty


def check_ci_ordering(df: pd.DataFrame, metrics: List[str] = METRIC_COLUMNS) -> pd.DataFrame:
    """Rows where lower > point or point > upper for any 95% CI present."""
    bad_rows = []
    for metric in metrics:
        lower_col, upper_col = f"{metric}_CI_lower", f"{metric}_CI_upper"
        if metric not in df.columns or lower_col not in df.columns or upper_col not in df.columns:
            continue
        sub = df[[metric, lower_col, upper_col]].dropna()
        violates = (sub[lower_col] > sub[metric] + 1e-9) | (sub[metric] > sub[upper_col] + 1e-9)
        if violates.any():
            bad_rows.append(df.loc[sub.index[violates]])
    if not bad_rows:
        return pd.DataFrame()
    return pd.concat(bad_rows)


def check_predictions_array(y_true: np.ndarray, y_pred: np.ndarray) -> List[str]:
    """Problems found in a (y_true, y_pred) pair; empty if clean."""
    problems = []
    if y_true.shape != y_pred.shape:
        problems.append(f"shape mismatch: y_true={y_true.shape} vs y_pred={y_pred.shape}")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        problems.append("contains NaN")
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        problems.append("contains Inf")
    if y_true.size == 0:
        problems.append("empty array")
    return problems


# --------------------------------------------------------------------------- #
# Synthetic fixtures: a clean result and corrupted variants
# --------------------------------------------------------------------------- #

def _make_clean_results_df(n_years: int = 5) -> pd.DataFrame:
    years = list(range(2020, 2020 + n_years))
    rows = []
    for cfg in ["M1", "M2", "M3", "M4", "M5"]:
        for y in years:
            rows.append({
                "fold": f"{cfg}_{y}", "test_year": y, "config": cfg, "model": "seq2seq",
                "NSE": 0.5, "KGE": 0.4, "HitRatio": 0.3, "FARate": 0.1, "FARatio": 0.2,
                "F1": 0.35, "Precision": 0.4, "Recall": 0.3,
                "NSE_CI_lower": 0.4, "NSE_CI_upper": 0.6,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# find_nan_inf
# --------------------------------------------------------------------------- #

def test_find_nan_inf_clean_dataframe_returns_empty():
    df = _make_clean_results_df()
    bad = find_nan_inf(df, METRIC_COLUMNS)
    assert bad.empty


def test_find_nan_inf_detects_nan():
    df = _make_clean_results_df()
    df.loc[2, "NSE"] = np.nan
    bad = find_nan_inf(df, METRIC_COLUMNS)
    assert len(bad) == 1
    assert bad.index[0] == 2


def test_find_nan_inf_detects_inf():
    df = _make_clean_results_df()
    df.loc[3, "F1"] = np.inf
    bad = find_nan_inf(df, METRIC_COLUMNS)
    assert len(bad) == 1
    assert bad.index[0] == 3


def test_find_nan_inf_detects_negative_inf():
    df = _make_clean_results_df()
    df.loc[0, "KGE"] = -np.inf
    bad = find_nan_inf(df, METRIC_COLUMNS)
    assert len(bad) == 1


def test_find_nan_inf_multiple_bad_rows():
    df = _make_clean_results_df()
    df.loc[1, "NSE"] = np.nan
    df.loc[4, "FARate"] = np.inf
    bad = find_nan_inf(df, METRIC_COLUMNS)
    assert set(bad.index) == {1, 4}


def test_find_nan_inf_ignores_absent_columns():
    df = _make_clean_results_df().drop(columns=["Precision", "Recall"])
    bad = find_nan_inf(df, METRIC_COLUMNS)  # Precision/Recall absent: must not fail
    assert bad.empty


def test_find_nan_inf_on_heterogeneous_schema_needs_row_filtering():
    """Regression: concatenating evaluation rows (NSE/KGE/...) with robustness
    rows (n_replicas/frac/...) yields legitimate NaNs in the columns each row
    type does not use, so find_nan_inf() must not be applied to the mixture
    without first filtering by row type (see
    test_real_results_consolidated_has_no_nan_inf_if_present)."""
    eval_df = _make_clean_results_df(n_years=1)
    robustness_df = pd.DataFrame([{
        "fold": "robustness_uniform_2024", "test_year": 2024,
        "config": "M5_robustness_uniform", "model": "seq2seq_base",
        "n_replicas": 50, "frac": 0.10,
        "mean_abs_deviation_from_baseline": 0.02,
        "mean_std_across_replicas": 0.15, "max_std_across_replicas": 0.6,
    }])
    mixed = pd.concat([eval_df, robustness_df], ignore_index=True)

    # Unfiltered: the robustness row shows up as "bad" (NaN in NSE/KGE/...).
    bad_unfiltered = find_nan_inf(mixed, METRIC_COLUMNS)
    assert len(bad_unfiltered) == 1
    assert bad_unfiltered.iloc[0]["config"] == "M5_robustness_uniform"

    # Filtered by row type (as the real integration test does), both subsets are clean.
    is_robustness = mixed["config"].str.contains("robustness", na=False)
    assert find_nan_inf(mixed.loc[~is_robustness], METRIC_COLUMNS).empty
    assert find_nan_inf(mixed.loc[is_robustness], ROBUSTNESS_COLUMNS).empty


# --------------------------------------------------------------------------- #
# check_required_columns
# --------------------------------------------------------------------------- #

def test_check_required_columns_all_present_returns_empty():
    df = _make_clean_results_df()
    assert check_required_columns(df) == []


def test_check_required_columns_detects_missing_column():
    df = _make_clean_results_df().drop(columns=["config"])
    missing = check_required_columns(df)
    assert "config" in missing


def test_check_required_columns_detects_all_nan_column():
    df = _make_clean_results_df()
    df["model"] = np.nan
    missing = check_required_columns(df)
    assert "model" in missing


# --------------------------------------------------------------------------- #
# check_ci_ordering
# --------------------------------------------------------------------------- #

def test_check_ci_ordering_valid_ci_returns_empty():
    df = _make_clean_results_df()
    bad = check_ci_ordering(df)
    assert bad.empty


def test_check_ci_ordering_detects_lower_above_point():
    df = _make_clean_results_df()
    df.loc[0, "NSE_CI_lower"] = 0.9  # > NSE=0.5, inconsistent
    bad = check_ci_ordering(df)
    assert len(bad) == 1


def test_check_ci_ordering_detects_upper_below_point():
    df = _make_clean_results_df()
    df.loc[0, "NSE_CI_upper"] = 0.1  # < NSE=0.5, inconsistent
    bad = check_ci_ordering(df)
    assert len(bad) == 1


def test_check_ci_ordering_missing_ci_columns_does_not_crash():
    df = _make_clean_results_df().drop(columns=["NSE_CI_lower", "NSE_CI_upper"])
    bad = check_ci_ordering(df)
    assert bad.empty


# --------------------------------------------------------------------------- #
# check_predictions_array
# --------------------------------------------------------------------------- #

def test_check_predictions_array_clean_pair_no_problems():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    assert check_predictions_array(y_true, y_pred) == []


def test_check_predictions_array_detects_shape_mismatch():
    problems = check_predictions_array(np.zeros(5), np.zeros(4))
    assert any("shape mismatch" in p for p in problems)


def test_check_predictions_array_detects_nan():
    problems = check_predictions_array(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    assert any("NaN" in p for p in problems)


def test_check_predictions_array_detects_inf():
    problems = check_predictions_array(np.array([1.0, 2.0]), np.array([1.0, np.inf]))
    assert any("Inf" in p for p in problems)


def test_check_predictions_array_detects_empty():
    problems = check_predictions_array(np.array([]), np.array([]))
    assert any("empty" in p for p in problems)


# --------------------------------------------------------------------------- #
# Integration: validate the real results if they exist (written by
# run_experiments_phase3.py); skipped if the experiment has not run yet.
# --------------------------------------------------------------------------- #

def test_real_results_consolidated_has_no_nan_inf_if_present():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} does not exist yet (run run_experiments_phase3.py first).")

    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        pytest.skip(f"{RESULTS_CSV} is empty.")

    # Heterogeneous schema: robustness rows (config contains "robustness") carry
    # their own dispersion columns instead of NSE/KGE/etc. NaN in METRIC_COLUMNS
    # for *those* rows is expected, not an integrity failure; they are validated
    # separately against ROBUSTNESS_COLUMNS.
    is_robustness = df["config"].str.contains("robustness", na=False)

    eval_rows = df.loc[~is_robustness]
    bad_eval = find_nan_inf(eval_rows, METRIC_COLUMNS)
    assert bad_eval.empty, f"Evaluation rows (M1-M5/baselines) with NaN/Inf in metrics:\n{bad_eval}"

    robustness_rows = df.loc[is_robustness]
    if not robustness_rows.empty:
        bad_robustness = find_nan_inf(robustness_rows, ROBUSTNESS_COLUMNS)
        assert bad_robustness.empty, f"Robustness rows with NaN/Inf in their columns:\n{bad_robustness}"


def test_real_results_consolidated_has_required_columns_if_present():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} does not exist yet.")

    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        pytest.skip(f"{RESULTS_CSV} is empty.")

    missing = check_required_columns(df)
    assert missing == [], f"Required columns missing or empty: {missing}"


def test_real_results_confidence_intervals_are_consistent_if_present():
    if not RESULTS_CSV.exists():
        pytest.skip(f"{RESULTS_CSV} does not exist yet.")

    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        pytest.skip(f"{RESULTS_CSV} is empty.")

    bad = check_ci_ordering(df)
    assert bad.empty, f"Rows with an inconsistent 95% CI (lower > point or point > upper):\n{bad}"


def test_real_prediction_arrays_have_no_nan_inf_if_present():
    if not PREDICTIONS_DIR.exists():
        pytest.skip(f"{PREDICTIONS_DIR} does not exist yet.")

    npz_files = sorted(PREDICTIONS_DIR.glob("*.npz"))
    if not npz_files:
        pytest.skip(f"No .npz files in {PREDICTIONS_DIR}.")

    all_problems = {}
    checked = 0
    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        if "y_true" not in data or "y_pred" not in data:
            continue  # robustness files (predictions/std/mean/baseline): different schema
        problems = check_predictions_array(data["y_true"].astype(float), data["y_pred"].astype(float))
        if problems:
            all_problems[f.name] = problems
        checked += 1

    assert not all_problems, f"Prediction integrity problems: {all_problems}"
    if checked == 0:
        pytest.skip("No .npz with y_true/y_pred found (robustness files only).")


def test_real_robustness_arrays_have_no_nan_inf_if_present():
    if not PREDICTIONS_DIR.exists():
        pytest.skip(f"{PREDICTIONS_DIR} does not exist yet.")

    npz_files = sorted(PREDICTIONS_DIR.glob("robustness_*.npz"))
    if not npz_files:
        pytest.skip("No robustness_*.npz files.")

    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        for key in ("predictions", "mean", "std", "baseline"):
            arr = data[key].astype(float)
            assert not np.isnan(arr).any(), f"{f.name}[{key}] contains NaN"
            assert not np.isinf(arr).any(), f"{f.name}[{key}] contains Inf"


# --------------------------------------------------------------------------- #
# Incremental checkpointing and automatic resume
# (phase3._save_results / phase3._is_fold_complete / phase3._load_existing_results)
# --------------------------------------------------------------------------- #

class _FakeLogger:
    """Avoids depending on init_logger()/real log files in these tests."""
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def exception(self, *a, **k): pass
    def error(self, *a, **k): pass


def _clean_row(fold: str, test_year: int, config: str = "M5", **extra) -> dict:
    row = {"fold": fold, "test_year": test_year, "hash": "deadbeef", "config": config, "model": "seq2seq_rf"}
    row.update({c: 0.5 for c in METRIC_COLUMNS})
    row.update(extra)
    return row


@pytest.fixture
def isolated_results_csv(tmp_path, monkeypatch):
    """Redirect phase3.RESULTS_CSV to a temporary file so the real one is never touched."""
    csv_path = tmp_path / "results_consolidated.csv"
    monkeypatch.setattr(phase3, "RESULTS_CSV", csv_path)
    return csv_path


def test_save_results_writes_new_file_incrementally(isolated_results_csv):
    logger = _FakeLogger()
    assert not isolated_results_csv.exists()

    phase3._save_results(_clean_row("M5_2014", 2014), logger)
    assert isolated_results_csv.exists()
    df = pd.read_csv(isolated_results_csv)
    assert len(df) == 1
    assert df.iloc[0]["fold"] == "M5_2014"

    # A second fold in a SEPARATE call (each fold is saved as soon as it is
    # computed, not in a batch at the end of the phase).
    phase3._save_results(_clean_row("M5_2015", 2015), logger)
    df = pd.read_csv(isolated_results_csv)
    assert len(df) == 2
    assert set(df["fold"]) == {"M5_2014", "M5_2015"}


def test_save_results_upserts_without_duplicating_primary_key(isolated_results_csv):
    """Recomputing a fold already present must REPLACE its row, not duplicate it."""
    logger = _FakeLogger()

    phase3._save_results(_clean_row("M5_2014", 2014, NSE=0.1), logger)
    phase3._save_results(_clean_row("M5_2014", 2014, NSE=0.9), logger)  # recomputation with another value

    df = pd.read_csv(isolated_results_csv)
    assert len(df) == 1  # not duplicated
    assert df.iloc[0]["fold"] == "M5_2014"
    assert df.iloc[0]["NSE"] == pytest.approx(0.9)  # the most recent value is kept


def test_save_results_is_atomic_no_tmp_file_left_behind(isolated_results_csv):
    logger = _FakeLogger()
    phase3._save_results(_clean_row("M5_2014", 2014), logger)
    tmp_path = isolated_results_csv.with_suffix(isolated_results_csv.suffix + ".tmp")
    assert not tmp_path.exists()  # already consumed by os.replace()


def test_save_results_accepts_list_of_rows(isolated_results_csv):
    logger = _FakeLogger()
    rows = [_clean_row("M5_2014", 2014), _clean_row("M5_2015", 2015)]
    phase3._save_results(rows, logger)
    df = pd.read_csv(isolated_results_csv)
    assert len(df) == 2


def test_save_results_empty_input_is_a_noop(isolated_results_csv):
    logger = _FakeLogger()
    phase3._save_results([], logger)
    assert not isolated_results_csv.exists()
    phase3._save_results([None], logger)  # list of None only
    assert not isolated_results_csv.exists()


def test_load_existing_results_none_when_missing(isolated_results_csv):
    assert phase3._load_existing_results() is None


def test_is_fold_complete_true_only_for_fully_populated_row(isolated_results_csv):
    logger = _FakeLogger()
    phase3._save_results(_clean_row("M5_2014", 2014), logger)
    existing = phase3._load_existing_results()

    assert phase3._is_fold_complete(existing, "M5_2014") is True
    assert phase3._is_fold_complete(existing, "M5_1999") is False  # absent


def test_is_fold_complete_false_when_a_metric_is_nan(isolated_results_csv):
    logger = _FakeLogger()
    row = _clean_row("M5_2014", 2014)
    row["NSE"] = float("nan")
    phase3._save_results(row, logger)
    existing = phase3._load_existing_results()

    assert phase3._is_fold_complete(existing, "M5_2014") is False


def test_is_fold_complete_uses_robustness_columns_for_robustness_rows(isolated_results_csv):
    logger = _FakeLogger()
    row = {
        "fold": "robustness_uniform_2025", "test_year": 2025, "config": "M5_robustness_uniform",
        "model": "seq2seq_base", "n_replicas": 50, "frac": 0.10,
        "mean_abs_deviation_from_baseline": 0.02, "mean_std_across_replicas": 0.1,
        "max_std_across_replicas": 0.5,
    }
    phase3._save_results(row, logger)
    existing = phase3._load_existing_results()

    assert phase3._is_fold_complete(existing, "robustness_uniform_2025", ROBUSTNESS_COLUMNS) is True
    # Under the evaluation-row criterion (METRIC_COLUMNS) it lacks those columns
    # -> considered incomplete, exactly as in the real resume.
    assert phase3._is_fold_complete(existing, "robustness_uniform_2025", METRIC_COLUMNS) is False


def test_resume_simulated_interruption_and_restart(isolated_results_csv):
    """Simulate a run interrupted in the middle of the year loop.

    On "resume" (re-reading the CSV and checking _is_fold_complete fold by
    fold, as run_cv/run_ablation/run_robustness do), completed folds must be
    skipped and only the missing ones recomputed.
    """
    logger = _FakeLogger()

    # "First pass" interrupted after completing 2014 and 2015 of a 4-year plan.
    years = [2014, 2015, 2016, 2017]
    for y in years[:2]:
        phase3._save_results(_clean_row(f"M5_{y}", y), logger)

    # "Resume": same loop as run_cv, deciding fold by fold.
    existing = phase3._load_existing_results()
    to_recompute = [y for y in years if not phase3._is_fold_complete(existing, f"M5_{y}")]

    assert to_recompute == [2016, 2017]

    # The missing folds are completed (each with its own atomic checkpoint, as in
    # the real loop) and the final CSV holds all 4 without duplicates.
    for y in to_recompute:
        phase3._save_results(_clean_row(f"M5_{y}", y), logger)

    final_df = pd.read_csv(isolated_results_csv)
    assert len(final_df) == 4
    assert set(final_df["fold"]) == {f"M5_{y}" for y in years}
    assert not final_df["fold"].duplicated().any()


def test_resume_disabled_would_recompute_everything():
    """--no_resume passes existing_df=None to _is_fold_complete throughout the
    loop (see run_cv/run_ablation/run_robustness); with existing_df=None no
    fold is considered complete, whatever its real state on disk."""
    fake_existing = pd.DataFrame([_clean_row("M5_2014", 2014)])
    assert phase3._is_fold_complete(fake_existing, "M5_2014") is True
    assert phase3._is_fold_complete(None, "M5_2014") is False
