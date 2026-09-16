"""Per-model metric registry used by :class:`Iteration` for cached per-model summaries.

The published fold metrics are computed by ``evaluation/metrics.py``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    import hydroeval as he
except Exception:
    he = None

def metric_overall_hydroeval(y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
    """Overall NSE and KGE through hydroeval (empty if hydroeval is unavailable).

    Args:
        y_true (np.ndarray): Observed values.
        y_pred (np.ndarray): Predicted values.
        **kwargs (Any): Ignored.

    Returns:
        Dict[str, Any]: ``{"overall": {"NSE", "KGE"}}``.
    """
    out = {"overall": {}}
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[m].ravel(), y_pred[m].ravel()
    if he is not None and yt.size and yp.size:
        try:
            out["overall"]["NSE"] = float(he.nse(yp, yt))
            out["overall"]["KGE"] = float(he.kge(yp, yt)[0][0])
        except Exception:
            pass
    return out

def metric_top10_hits_misses(y_true: np.ndarray, y_pred: np.ndarray, threshold: Optional[float] = None, **kwargs: Any) -> Dict[str, Any]:
    """Extreme subset: hits (pred >= obs), misses and ratios.

    Args:
        y_true (np.ndarray): Observed values.
        y_pred (np.ndarray): Predicted values.
        threshold (Optional[float]): Extreme threshold (training p90 in the pipeline).
        **kwargs (Any): Ignored.

    Returns:
        Dict[str, Any]: ``{"top10": {threshold, n, hits, misses, hit_ratio, miss_ratio}}``.
    """
    out = {"top10": {}}
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[m].ravel(), y_pred[m].ravel()
    if yt.size == 0:
        return out

    thr = threshold if threshold is not None else float(np.percentile(yt, 90.0))
    mask = yt >= thr
    n = int(mask.sum())
    hits = int(np.sum(yp[mask] >= yt[mask])) if n > 0 else 0
    misses = int(n - hits)
    out["top10"].update({
        "threshold": thr,
        "n": n,
        "hits": hits,
        "misses": misses,
        "hit_ratio": float(hits / n) if n > 0 else None,
        "miss_ratio": float(misses / n) if n > 0 else None,
    })
    return out

def metric_alarm_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: Optional[float] = None, **kwargs: Any) -> Dict[str, Any]:
    """Operational alarm metrics: precision, recall (hit ratio), false alarm ratio and F1.

    Leakage is avoided by passing the fixed training threshold.

    Args:
        y_true (np.ndarray): Observed values.
        y_pred (np.ndarray): Predicted values.
        threshold (Optional[float]): Alarm threshold.
        **kwargs (Any): Ignored.

    Returns:
        Dict[str, Any]: ``Precision``, ``Recall_HitRatio``, ``FAR``, ``F1_Score``
        and ``mae_extremes`` (MAE on the observed extremes).
    """
    out = {}
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[m].ravel(), y_pred[m].ravel()

    if yt.size == 0:
        return out

    thr = threshold if threshold is not None else float(np.percentile(yt, 90.0))

    # Event = inflow above the alarm threshold
    actual_event = yt >= thr
    predicted_event = yp >= thr

    TP = np.sum(actual_event & predicted_event)
    FP = np.sum(~actual_event & predicted_event)
    FN = np.sum(actual_event & ~predicted_event)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    far = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    out["Precision"] = float(precision)
    out["Recall_HitRatio"] = float(recall)
    out["FAR"] = float(far)
    out["F1_Score"] = float(f1)

    # MAE restricted to the observed extremes
    if np.any(actual_event):
        out["mae_extremes"] = float(np.mean(np.abs(yt[actual_event] - yp[actual_event])))

    return out

def metric_timing_error(y_true: np.ndarray, y_pred: np.ndarray, threshold: Optional[float] = None, **kwargs: Any) -> Dict[str, Any]:
    """Mean absolute lead/lag error of the predicted peaks.

    For each observed peak above the threshold, the predicted maximum within
    ±3 steps is located and the absolute offset is averaged.

    Args:
        y_true (np.ndarray): Observed values.
        y_pred (np.ndarray): Predicted values.
        threshold (Optional[float]): Peak height threshold.
        **kwargs (Any): Ignored.

    Returns:
        Dict[str, Any]: ``{"mean_absolute_timing_error": float | None}``.
    """
    out = {"mean_absolute_timing_error": None}
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[m].ravel(), y_pred[m].ravel()

    if yt.size < 2:
        return out

    thr = threshold if threshold is not None else float(np.percentile(yt, 90.0))

    try:
        from scipy.signal import find_peaks
        # Observed peaks above the threshold
        obs_peaks, _ = find_peaks(yt, height=thr, distance=3)

        if len(obs_peaks) == 0:
            return out

        timing_errors = []
        window = 3  # search window of ±3 steps around each peak

        for p in obs_peaks:
            start_idx = max(0, p - window)
            end_idx = min(len(yp), p + window + 1)

            if start_idx < end_idx:
                pred_peak_relative = np.argmax(yp[start_idx:end_idx])
                pred_peak_abs = start_idx + pred_peak_relative
                timing_errors.append(abs(p - pred_peak_abs))

        if timing_errors:
            out["mean_absolute_timing_error"] = float(np.mean(timing_errors))

    except ImportError:
        pass  # scipy unavailable: timing error left as None

    return out


# Interchangeable registry read by Iteration
METRICS_REGISTRY = {
    "overall_hydroeval": metric_overall_hydroeval,
    "top10_hits_misses": metric_top10_hits_misses,
    "alarm_metrics": metric_alarm_metrics,
    "timing_error": metric_timing_error,
}