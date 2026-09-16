"""Evaluation metrics for extreme events, peak timing error and bootstrap
confidence intervals.

Complements ``pipeline/metrics.py`` (the registry used by ``Iteration``)
without depending on it, so that it can be reused independently by tests.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

try:
    from scipy.signal import find_peaks
except Exception:  # pragma: no cover - scipy is a project dependency
    find_peaks = None


def _clean_pair(y_true: ArrayLike, y_pred: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten both series and drop positions where either is NaN.

    Args:
        y_true (ArrayLike): Observed values.
        y_pred (ArrayLike): Predicted values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The cleaned ``(y_true, y_pred)`` pair.

    Raises:
        ValueError: If both series do not have the same length.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask], y_pred[mask]


def _safe_div(num: float, den: float) -> float:
    """Divide, returning ``0.0`` when the denominator is not positive.

    Args:
        num (float): Numerator.
        den (float): Denominator.

    Returns:
        float: ``num / den``, or ``0.0``.
    """
    return float(num / den) if den > 0 else 0.0


def _resolve_threshold_p90(y_true: np.ndarray, threshold_p90: Optional[float], caller: str) -> float:
    """Return the extreme-event threshold.

    A threshold fixed on the training partition is returned unchanged. If it
    is missing, the 90th percentile of ``y_true`` (the TEST series) is used
    and a warning is emitted, because that leaks test information; this is
    only acceptable in tests or exploration. Production code passes the p90
    fixed on training (see ``Iteration._norm_params['Qe']['p90']``).

    Args:
        y_true (np.ndarray): Observed values.
        threshold_p90 (Optional[float]): Training threshold, if available.
        caller (str): Name of the calling function, used in the warning.

    Returns:
        float: The threshold.
    """
    if threshold_p90 is not None:
        return float(threshold_p90)
    warnings.warn(
        f"{caller}: threshold_p90 not specified, computing it on the TEST y_true "
        "(data leakage). Only valid for tests/exploration; in production pass the p90 "
        "fixed on training (see Iteration._norm_params['Qe']['p90']).",
        RuntimeWarning,
        stacklevel=3,
    )
    return float(np.percentile(y_true, 90.0))


# --------------------------------------------------------------------------- #
# 1. Extreme-event metrics
# --------------------------------------------------------------------------- #

def compute_extreme_metrics(y_true: ArrayLike, y_pred: ArrayLike, threshold_p90: Optional[float] = None) -> Dict[str, float]:
    """Operational metrics for the event "inflow above the threshold".

    - HitRatio: among real extreme events, fraction where the prediction
      reached or exceeded the observed value (``y_pred >= y_true``).
    - FARate = FP / (FP + TN): false alarm rate over real negatives.
    - FARatio = FP / (TP + FP): fraction of raised alarms that were false.
    - Precision, Recall and F1 of the binary classification
      "value >= threshold".

    Args:
        y_true (ArrayLike): Observed values.
        y_pred (ArrayLike): Predicted values.
        threshold_p90 (Optional[float]): Threshold fixed on training. If
            ``None``, the 90th percentile of ``y_true`` is used (tests and
            exploration only; it leaks test information).

    Returns:
        Dict[str, float]: ``threshold``, ``n_extreme``, ``TP``, ``FP``, ``FN``,
        ``TN``, ``HitRatio``, ``FARate``, ``FARatio``, ``Precision``,
        ``Recall`` and ``F1``.

    Raises:
        ValueError: If both series do not have the same length.
    """
    y_true, y_pred = _clean_pair(y_true, y_pred)

    if y_true.size == 0:
        return {
            "threshold": float(threshold_p90) if threshold_p90 is not None else None,
            "n_extreme": 0, "TP": 0, "FP": 0, "FN": 0, "TN": 0,
            "HitRatio": 0.0, "FARate": 0.0, "FARatio": 0.0,
            "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
        }

    threshold_p90 = _resolve_threshold_p90(y_true, threshold_p90, "compute_extreme_metrics")

    actual_event = y_true >= threshold_p90
    predicted_event = y_pred >= threshold_p90

    TP = int(np.sum(actual_event & predicted_event))
    FP = int(np.sum(~actual_event & predicted_event))
    FN = int(np.sum(actual_event & ~predicted_event))
    TN = int(np.sum(~actual_event & ~predicted_event))

    precision = _safe_div(TP, TP + FP)
    recall = _safe_div(TP, TP + FN)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    fa_rate = _safe_div(FP, FP + TN)
    fa_ratio = _safe_div(FP, TP + FP)

    n_extreme = int(np.sum(actual_event))
    if n_extreme > 0:
        hit_ratio = float(np.sum(y_pred[actual_event] >= y_true[actual_event]) / n_extreme)
    else:
        hit_ratio = 0.0

    return {
        "threshold": float(threshold_p90),
        "n_extreme": n_extreme,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "HitRatio": hit_ratio,
        "FARate": fa_rate,
        "FARatio": fa_ratio,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


# --------------------------------------------------------------------------- #
# 2. Peak timing error (lead/lag) on matched peaks
# --------------------------------------------------------------------------- #

def peak_timing_error(y_true: ArrayLike, y_pred: ArrayLike, threshold_p90: Optional[float] = None, search_window: int = 3) -> Dict:
    """Match observed peaks to predicted peaks and measure their lag in days.

    Each observed peak above ``threshold_p90`` is matched with the local
    maximum of the prediction within ``+/- search_window`` steps.
    ``lag > 0`` means the peak is predicted late, ``lag < 0`` early, and
    ``lag == 0`` on the exact day.

    Args:
        y_true (ArrayLike): Observed values.
        y_pred (ArrayLike): Predicted values.
        threshold_p90 (Optional[float]): Threshold fixed on training.
        search_window (int): Half-width of the matching window, in steps.

    Returns:
        Dict: ``mean_lag`` (signed), ``mean_absolute_lag``, ``n_peaks``,
        ``lags`` and ``threshold``. Both means are ``NaN`` when no peak is
        found.

    Raises:
        ImportError: If scipy is not installed.
        ValueError: If both series do not have the same length.
    """
    y_true, y_pred = _clean_pair(y_true, y_pred)
    # NaN rather than None keeps the CSV column numeric (a None turns it into
    # an object column and breaks cross-fold means/std). Safe for resume:
    # METRIC_COLUMNS contains no PeakTiming_* column, so a NaN here never
    # marks a fold as incomplete in _is_fold_complete.
    out = {"mean_lag": np.nan, "mean_absolute_lag": np.nan, "n_peaks": 0, "lags": [], "threshold": None}

    if y_true.size < 2:
        return out
    if find_peaks is None:
        raise ImportError("scipy is required for peak_timing_error (scipy.signal.find_peaks).")

    threshold_p90 = _resolve_threshold_p90(y_true, threshold_p90, "peak_timing_error")
    out["threshold"] = float(threshold_p90)

    true_peaks, _ = find_peaks(y_true, height=threshold_p90, distance=max(1, search_window))
    if true_peaks.size == 0:
        return out

    lags: List[int] = []
    for idx in true_peaks:
        start = max(0, idx - search_window)
        end = min(y_pred.size, idx + search_window + 1)
        if start >= end:
            continue
        local_peak = start + int(np.argmax(y_pred[start:end]))
        lags.append(int(local_peak - idx))

    if not lags:
        return out

    lags_arr = np.asarray(lags, dtype=float)
    out.update({
        "mean_lag": float(np.mean(lags_arr)),
        "mean_absolute_lag": float(np.mean(np.abs(lags_arr))),
        "n_peaks": len(lags),
        "lags": lags,
    })
    return out


# --------------------------------------------------------------------------- #
# 3. NSE / KGE (self-contained, no hydroeval dependency) + bootstrap CI
# --------------------------------------------------------------------------- #

def nse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Nash-Sutcliffe Efficiency.

    Args:
        y_true (ArrayLike): Observed values.
        y_pred (ArrayLike): Predicted values.

    Returns:
        float: NSE, or ``NaN`` for an empty or constant observed series.
    """
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def kge(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Kling-Gupta Efficiency (Gupta et al., 2009).

    Args:
        y_true (ArrayLike): Observed values.
        y_pred (ArrayLike): Predicted values.

    Returns:
        float: KGE, or ``NaN`` when it is undefined (empty series, zero
        variance or zero mean).
    """
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if y_true.size == 0 or np.std(y_true) == 0 or np.mean(y_true) == 0:
        return float("nan")
    r = float(np.corrcoef(y_pred, y_true)[0, 1]) if np.std(y_pred) > 0 else float("nan")
    alpha = float(np.std(y_pred) / np.std(y_true))
    beta = float(np.mean(y_pred) / np.mean(y_true))
    if np.isnan(r):
        return float("nan")
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def _f1_metric(y_true: ArrayLike, y_pred: ArrayLike, threshold_p90: Optional[float] = None) -> float:
    """F1 score of :func:`compute_extreme_metrics`.

    Args:
        y_true (ArrayLike): Observed values.
        y_pred (ArrayLike): Predicted values.
        threshold_p90 (Optional[float]): Threshold fixed on training.

    Returns:
        float: F1 score.
    """
    return compute_extreme_metrics(y_true, y_pred, threshold_p90=threshold_p90)["F1"]


def _hit_ratio_metric(y_true: ArrayLike, y_pred: ArrayLike, threshold_p90: Optional[float] = None) -> float:
    """HitRatio of :func:`compute_extreme_metrics`.

    Args:
        y_true (ArrayLike): Observed values.
        y_pred (ArrayLike): Predicted values.
        threshold_p90 (Optional[float]): Threshold fixed on training.

    Returns:
        float: Hit ratio.
    """
    return compute_extreme_metrics(y_true, y_pred, threshold_p90=threshold_p90)["HitRatio"]


def compute_bootstrap_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Optional[float]]:
    """Percentile bootstrap confidence interval for a generic metric.

    Resamples ``(y_true, y_pred)`` pairs with replacement ``n_bootstraps``
    times.

    Args:
        y_true (ArrayLike): Observed values.
        y_pred (ArrayLike): Predicted values.
        metric_fn (Callable[[np.ndarray, np.ndarray], float]): Metric to bootstrap.
        n_bootstraps (int): Number of resamples.
        alpha (float): Significance level; the interval covers ``1 - alpha``.
        seed (int): Seed of the resampling generator.

    Returns:
        Dict[str, Optional[float]]: ``point``, ``lower``, ``upper``, ``alpha``,
        ``n_bootstraps`` and ``n_valid``. Bounds are ``None`` when the series
        is empty or no resample is valid.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = y_true.shape[0]

    if n == 0:
        return {"point": None, "lower": None, "upper": None, "alpha": alpha, "n_bootstraps": n_bootstraps}

    point = float(metric_fn(y_true, y_pred))

    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_bootstraps, dtype=float)
    for b in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        try:
            v = float(metric_fn(y_true[idx], y_pred[idx]))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            # Degenerate resample (e.g. zero variance) -> NaN, excluded from
            # the percentiles. Any other error (a real bug in metric_fn)
            # propagates instead of being silently masked across resamples.
            v = np.nan
        boot_vals[b] = v

    valid = boot_vals[~np.isnan(boot_vals)]
    if valid.size == 0:
        return {"point": point, "lower": None, "upper": None, "alpha": alpha,
                "n_bootstraps": n_bootstraps, "n_valid": 0}

    lower = float(np.percentile(valid, 100 * (alpha / 2)))
    upper = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "alpha": alpha,
        "n_bootstraps": n_bootstraps,
        "n_valid": int(valid.size),
    }


def bootstrap_ci_standard_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    threshold_p90: Optional[float] = None,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Bootstrap confidence intervals for NSE, KGE, F1 and HitRatio.

    Every metric uses the same seed, so resamples are independent across
    metrics but reproducible across calls.

    Args:
        y_true (ArrayLike): Observed values.
        y_pred (ArrayLike): Predicted values.
        threshold_p90 (Optional[float]): Threshold fixed on training for F1 and
            HitRatio.
        n_bootstraps (int): Number of resamples per metric.
        alpha (float): Significance level.
        seed (int): Seed of the resampling generator.

    Returns:
        Dict[str, Dict[str, Optional[float]]]: One
        :func:`compute_bootstrap_ci` result per metric.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    if threshold_p90 is None and y_true.size:
        threshold_p90 = _resolve_threshold_p90(y_true, threshold_p90, "bootstrap_ci_standard_metrics")

    return {
        "NSE": compute_bootstrap_ci(y_true, y_pred, nse, n_bootstraps, alpha, seed),
        "KGE": compute_bootstrap_ci(y_true, y_pred, kge, n_bootstraps, alpha, seed),
        "F1": compute_bootstrap_ci(
            y_true, y_pred, lambda yt, yp: _f1_metric(yt, yp, threshold_p90), n_bootstraps, alpha, seed
        ),
        "HitRatio": compute_bootstrap_ci(
            y_true, y_pred, lambda yt, yp: _hit_ratio_metric(yt, yp, threshold_p90), n_bootstraps, alpha, seed
        ),
    }
