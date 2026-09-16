"""Synthetic cases for resultados/prueba63/src/evaluation/metrics.py.

Perfect predictions, null predictions, all-zero series (division by zero),
peaks shifted by exactly 1 day, and bootstrap confidence intervals.
"""
from __future__ import annotations

import numpy as np
import pytest

from resultados.prueba63.src.evaluation.metrics import (
    compute_extreme_metrics,
    peak_timing_error,
    compute_bootstrap_ci,
    bootstrap_ci_standard_metrics,
    nse,
    kge,
)


# --------------------------------------------------------------------------- #
# compute_extreme_metrics
# --------------------------------------------------------------------------- #

def _series_with_extremes(n=100, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 10, size=n)
    # force a clear 10% of extreme events above the rest
    peak_idx = rng.choice(n, size=max(1, n // 10), replace=False)
    base[peak_idx] += 50.0
    return base


def test_compute_extreme_metrics_perfect_predictions():
    y_true = _series_with_extremes()
    y_pred = y_true.copy()

    out = compute_extreme_metrics(y_true, y_pred)

    assert out["Precision"] == pytest.approx(1.0)
    assert out["Recall"] == pytest.approx(1.0)
    assert out["F1"] == pytest.approx(1.0)
    assert out["HitRatio"] == pytest.approx(1.0)
    assert out["FARate"] == pytest.approx(0.0)
    assert out["FARatio"] == pytest.approx(0.0)
    assert out["FP"] == 0
    assert out["FN"] == 0


def test_compute_extreme_metrics_null_predictions():
    y_true = _series_with_extremes()
    y_pred = np.zeros_like(y_true)  # never predicts anything above the threshold

    out = compute_extreme_metrics(y_true, y_pred)

    assert out["TP"] == 0
    assert out["FP"] == 0
    assert out["Precision"] == 0.0  # 0/0 division -> 0.0, no NaN/crash
    assert out["Recall"] == 0.0
    assert out["F1"] == 0.0
    assert out["HitRatio"] == 0.0
    assert out["FARate"] == 0.0
    assert out["FARatio"] == 0.0  # also 0/0 -> 0.0


def test_compute_extreme_metrics_absolute_zeros_no_division_errors():
    y_true = np.zeros(20)
    y_pred = np.zeros(20)

    out = compute_extreme_metrics(y_true, y_pred)  # threshold_p90 = percentile(zeros,90) = 0

    assert np.isfinite(out["Precision"])
    assert np.isfinite(out["Recall"])
    assert np.isfinite(out["F1"])
    assert np.isfinite(out["HitRatio"])
    assert np.isfinite(out["FARate"])
    assert np.isfinite(out["FARatio"])
    # y_true >= 0 and y_pred >= 0 hold for every sample -> TP = N
    assert out["TP"] == 20
    assert out["Precision"] == pytest.approx(1.0)


def test_compute_extreme_metrics_empty_input_does_not_crash():
    out = compute_extreme_metrics(np.array([]), np.array([]))
    assert out["n_extreme"] == 0
    assert out["Precision"] == 0.0
    assert out["F1"] == 0.0


def test_compute_extreme_metrics_explicit_threshold_avoids_leakage():
    y_true = np.array([1.0, 2.0, 3.0, 100.0])
    y_pred = np.array([1.0, 2.0, 3.0, 100.0])
    # Externally fixed threshold (e.g. computed on train), not derived from this y_true.
    out = compute_extreme_metrics(y_true, y_pred, threshold_p90=50.0)
    assert out["threshold"] == 50.0
    assert out["n_extreme"] == 1


# --------------------------------------------------------------------------- #
# peak_timing_error
# --------------------------------------------------------------------------- #

def _single_peak_series(n=40, peak_idx=20, height=100.0, base=1.0):
    y = np.full(n, base, dtype=float)
    # triangular shape so that find_peaks detects a single clear maximum
    for offset in range(-5, 6):
        idx = peak_idx + offset
        if 0 <= idx < n:
            y[idx] = max(y[idx], height - abs(offset) * 15.0)
    return y


def test_peak_timing_error_exact_one_day_lag():
    y_true = _single_peak_series(peak_idx=20)
    y_pred = _single_peak_series(peak_idx=21)  # peak shifted by +1 day

    out = peak_timing_error(y_true, y_pred, threshold_p90=50.0, search_window=3)

    assert out["n_peaks"] == 1
    assert out["mean_lag"] == pytest.approx(1.0)
    assert out["mean_absolute_lag"] == pytest.approx(1.0)


def test_peak_timing_error_perfect_alignment_zero_lag():
    y_true = _single_peak_series(peak_idx=20)
    y_pred = y_true.copy()

    out = peak_timing_error(y_true, y_pred, threshold_p90=50.0, search_window=3)

    assert out["n_peaks"] == 1
    assert out["mean_lag"] == pytest.approx(0.0)
    assert out["mean_absolute_lag"] == pytest.approx(0.0)


def test_peak_timing_error_no_peaks_above_threshold():
    y_true = np.full(30, 1.0)
    y_pred = np.full(30, 1.0)

    out = peak_timing_error(y_true, y_pred, threshold_p90=50.0)

    assert out["n_peaks"] == 0
    # NaN rather than None keeps the CSV column numeric, so across-fold
    # means/std of the paired analysis do not break.
    assert np.isnan(out["mean_lag"])
    assert np.isnan(out["mean_absolute_lag"])


def test_peak_timing_error_too_short_series():
    out = peak_timing_error(np.array([1.0]), np.array([1.0]))
    assert out["n_peaks"] == 0
    assert np.isnan(out["mean_lag"])
    assert np.isnan(out["mean_absolute_lag"])


# --------------------------------------------------------------------------- #
# compute_bootstrap_ci / bootstrap_ci_standard_metrics
# --------------------------------------------------------------------------- #

def test_compute_bootstrap_ci_constant_metric_collapses_ci():
    y_true = _series_with_extremes()
    y_pred = y_true.copy()

    out = compute_bootstrap_ci(y_true, y_pred, metric_fn=lambda yt, yp: 0.42, n_bootstraps=50, seed=1)

    assert out["point"] == pytest.approx(0.42)
    assert out["lower"] == pytest.approx(0.42)
    assert out["upper"] == pytest.approx(0.42)


def test_compute_bootstrap_ci_is_deterministic_given_seed():
    y_true = _series_with_extremes(seed=1)
    y_pred = y_true + np.random.default_rng(2).normal(0, 1.0, size=y_true.shape)

    out1 = compute_bootstrap_ci(y_true, y_pred, metric_fn=nse, n_bootstraps=100, seed=7)
    out2 = compute_bootstrap_ci(y_true, y_pred, metric_fn=nse, n_bootstraps=100, seed=7)

    assert out1 == out2


def test_compute_bootstrap_ci_bounds_bracket_point_for_perfect_fit():
    y_true = _series_with_extremes()
    y_pred = y_true.copy()  # perfect NSE = 1.0 in every resample

    out = compute_bootstrap_ci(y_true, y_pred, metric_fn=nse, n_bootstraps=100, seed=3)

    assert out["point"] == pytest.approx(1.0)
    assert out["lower"] <= out["point"] <= out["upper"]


def test_compute_bootstrap_ci_empty_input():
    out = compute_bootstrap_ci(np.array([]), np.array([]), metric_fn=nse)
    assert out["point"] is None
    assert out["lower"] is None
    assert out["upper"] is None


def test_bootstrap_ci_standard_metrics_has_all_four_keys():
    y_true = _series_with_extremes(seed=4)
    y_pred = y_true + np.random.default_rng(5).normal(0, 2.0, size=y_true.shape)

    out = bootstrap_ci_standard_metrics(y_true, y_pred, n_bootstraps=50, seed=9)

    assert set(out.keys()) == {"NSE", "KGE", "F1", "HitRatio"}
    for key, ci in out.items():
        assert ci["point"] is not None, f"{key} point estimate should not be None"
        assert ci["lower"] <= ci["upper"]


# --------------------------------------------------------------------------- #
# nse / kge sanity
# --------------------------------------------------------------------------- #

def test_nse_perfect_fit_is_one():
    y = _series_with_extremes()
    assert nse(y, y) == pytest.approx(1.0)


def test_kge_perfect_fit_is_one():
    y = _series_with_extremes()
    assert kge(y, y) == pytest.approx(1.0)


def test_nse_constant_series_returns_nan():
    y_true = np.full(10, 5.0)
    y_pred = np.full(10, 5.0)
    assert np.isnan(nse(y_true, y_pred))
