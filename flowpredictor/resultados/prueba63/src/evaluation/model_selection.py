"""Comparison and selection of the configurations M1-M5 from results_consolidated.csv.

Evidence is combined across the cross-validation strategies present
(expanding and LOYO, distinguished by the ``strategy`` column).

Two-layer design:

- :func:`select_best_model`: selector layer. It computes no metric; it only
  calls the injected decision function and picks the highest-scoring
  configuration.
- :func:`lexicographic_constrained_score`: default decision function
  (constrained hierarchical selection, as in the hydrological model-selection
  literature: maximize delta Hit Ratio subject to NSE > 0 and FARate <= alpha).
  :func:`weighted_normalized_score` (normalized weighted sum of delta Hit Ratio,
  peak timing error and NSE) is kept as an interchangeable alternative.

:func:`build_final_report` joins the selection of M* with the metrics of the
real holdout (``strategy="holdout_final"``) in a single report.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

M1_M5 = ["M1", "M2", "M3", "M4", "M5"]
BASELINE_CONFIGS = ["quantile_regression", "gradient_boosting_quantile", "heuristic_ensemble"]
_DECISION_METRICS = ["DeltaHitRatio", "PeakTiming_mean_absolute_lag", "NSE", "FARate"]


def compute_delta_hit_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``DeltaHitRatio`` to the M1-M5 rows.

    ``DeltaHitRatio = HitRatio(MX) - mean HitRatio of the baselines`` in the
    SAME fold (same ``test_year`` and ``strategy``); the baseline reference is
    the mean of quantile_regression, gradient_boosting_quantile and
    heuristic_ensemble.

    Args:
        df (pd.DataFrame): Consolidated results.

    Returns:
        pd.DataFrame: The M1-M5 rows with ``BaselineHitRatio`` and ``DeltaHitRatio``.
    """
    baseline_hr = (
        df[df["config"].isin(BASELINE_CONFIGS)]
        .groupby(["test_year", "strategy"])["HitRatio"]
        .mean()
        .rename("BaselineHitRatio")
    )
    m5_rows = df[df["config"].isin(M1_M5)].copy()
    m5_rows = m5_rows.join(baseline_hr, on=["test_year", "strategy"])
    m5_rows["DeltaHitRatio"] = m5_rows["HitRatio"] - m5_rows["BaselineHitRatio"]
    return m5_rows


HOLDOUT_STRATEGY = "holdout_final"

# Published variant (seed / threshold percentile). Literals on purpose, as in
# run_experiments_phase3.py and pipeline/iteration.py.
PUBLISHED_SEED = 92
PUBLISHED_THRESHOLD_PCT = 90


def filter_published_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the rows of the published variant (seed 92, p90).

    Sensitivity variants (another seed, p95 threshold) share ``config``,
    ``strategy`` and ``horizon`` with the published rows and differ only in
    ``seed``/``threshold_percentile`` (and the ``fold`` suffix). Without this
    filter they would be silently averaged together. Rows without those
    columns (older CSVs) count as the published variant.

    Args:
        df (pd.DataFrame): Consolidated results.

    Returns:
        pd.DataFrame: Filtered rows.
    """
    if "seed" in df.columns:
        df = df[df["seed"].fillna(PUBLISHED_SEED) == PUBLISHED_SEED]
    if "threshold_percentile" in df.columns:
        df = df[df["threshold_percentile"].fillna(PUBLISHED_THRESHOLD_PCT) == PUBLISHED_THRESHOLD_PCT]
    return df


def aggregate_config_metrics(df: pd.DataFrame, horizon: str = "48h",
                              exclude_holdout: bool = True) -> Dict[str, dict]:
    """Per-configuration mean/std/number of folds of the decision metrics.

    For each of M1-M5, the statistics of ``DeltaHitRatio``,
    ``PeakTiming_mean_absolute_lag``, ``NSE`` and ``FARate`` are computed three
    times: ``expanding`` and ``loyo`` separately (for reporting) and
    ``combined`` (all strategies pooled, the evidence that feeds the selection
    score).

    ``compute_delta_hit_ratio`` groups by ``(test_year, strategy)`` without
    distinguishing the horizon, so rows are first restricted to a single
    horizon (``"48h"`` by default); otherwise Hit Ratios of different horizons
    would be silently averaged. A ``df`` without a ``horizon`` column
    (synthetic test DataFrames) is not filtered.

    Args:
        df (pd.DataFrame): Consolidated results.
        horizon (str): Horizon to keep.
        exclude_holdout (bool): Drop the blind-holdout rows.

    Returns:
        Dict[str, dict]: ``{config: {strategy: {metric statistics}}}``.
    """
    if "horizon" in df.columns:
        df = df[df["horizon"] == horizon]
    # 'combined' pools every strategy present; without this filter the blind
    # holdout would enter the evidence that selects M* and the means reported
    # as "cross-validation folds". It also equalizes the fold set across
    # configurations: M1/M2/M5 have a holdout row and M3/M4 do not, so without
    # filtering 18 folds would be compared against 17.
    if exclude_holdout and "strategy" in df.columns:
        df = df[df["strategy"] != HOLDOUT_STRATEGY]
    df = filter_published_variant(df)
    rows = compute_delta_hit_ratio(df)
    out: Dict[str, dict] = {}
    for cfg in M1_M5:
        sub = rows[rows["config"] == cfg]
        entry: Dict[str, dict] = {}
        for strat in ("expanding", "loyo", "combined"):
            part = sub if strat == "combined" else sub[sub["strategy"] == strat]
            stats: Dict[str, Optional[float]] = {"n_folds": int(len(part))}
            for m in _DECISION_METRICS:
                has_data = m in part.columns and not part[m].isna().all()
                stats[f"{m}_mean"] = float(part[m].mean()) if has_data else None
                stats[f"{m}_std"] = float(part[m].std()) if has_data else None
            entry[strat] = stats
        out[cfg] = entry
    return out


def weighted_normalized_score(configs_metrics: Dict[str, dict],
                               weights: tuple = (1 / 3, 1 / 3, 1 / 3)) -> Dict[str, float]:
    """Alternative decision function: normalized weighted sum.

    ``score(MX) = w1*norm(dHitRatio) + w2*(1 - norm(PeakTimingError)) + w3*norm(NSE)``,
    with min-max normalization over the configurations present, on the
    combined evidence.

    Args:
        configs_metrics (Dict[str, dict]): Output of :func:`aggregate_config_metrics`.
        weights (tuple): ``(w1, w2, w3)``.

    Returns:
        Dict[str, float]: Score per configuration.
    """
    w_hr, w_pt, w_nse = weights
    configs = list(configs_metrics.keys())

    def _series(key: str) -> np.ndarray:
        return np.array([configs_metrics[c]["combined"][key] for c in configs], dtype=float)

    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if hi - lo == 0:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    hr_n = _minmax(_series("DeltaHitRatio_mean"))
    pt_n = _minmax(_series("PeakTiming_mean_absolute_lag_mean"))
    nse_n = _minmax(_series("NSE_mean"))

    scores = w_hr * hr_n + w_pt * (1 - pt_n) + w_nse * nse_n
    return {c: float(s) for c, s in zip(configs, scores)}


def lexicographic_constrained_score(
    configs_metrics: Dict[str, dict], nse_min: float = 0.0, farate_max: float = 0.10,
) -> Dict[str, float]:
    """Default decision function: constrained lexicographic selection.

    Maximize delta Hit Ratio subject to ``NSE > nse_min`` and
    ``FARate <= farate_max``, on the combined evidence. Configurations that
    violate a constraint score ``-inf`` (they can never win); among the
    feasible ones the score is their mean delta Hit Ratio (higher is better).

    Args:
        configs_metrics (Dict[str, dict]): Output of :func:`aggregate_config_metrics`.
        nse_min (float): Strict lower bound on mean NSE.
        farate_max (float): Upper bound on mean false alarm rate.

    Returns:
        Dict[str, float]: Score per configuration.

    Raises:
        RuntimeError: If no configuration satisfies the constraints.
    """
    scores: Dict[str, float] = {}
    for cfg, m in configs_metrics.items():
        c = m["combined"]
        nse, farate, delta_hr = c.get("NSE_mean"), c.get("FARate_mean"), c.get("DeltaHitRatio_mean")
        if nse is None or farate is None or delta_hr is None:
            scores[cfg] = float("-inf")
        elif nse > nse_min and farate <= farate_max:
            scores[cfg] = delta_hr
        else:
            scores[cfg] = float("-inf")

    if all(s == float("-inf") for s in scores.values()):
        raise RuntimeError(
            f"[SELECT] Ninguna config cumple NSE>{nse_min} y FARate<={farate_max} -- "
            f"no hay candidato válido para la selección lexicográfica."
        )
    return scores


def select_best_model(
    configs_metrics: Dict[str, dict],
    decision_fn: Callable[[Dict[str, dict]], Dict[str, float]] = lexicographic_constrained_score,
) -> Dict:
    """Selector layer: call ``decision_fn`` and pick the highest-scoring configuration.

    Args:
        configs_metrics (Dict[str, dict]): Output of :func:`aggregate_config_metrics`.
        decision_fn (Callable[[Dict[str, dict]], Dict[str, float]]): Decision function.

    Returns:
        Dict: ``selected``, ``decision_fn``, ``scores`` and ``metrics_by_config``.
    """
    scores = decision_fn(configs_metrics)
    best_config = max(scores, key=scores.get)
    return {
        "selected": best_config,
        "decision_fn": getattr(decision_fn, "__name__", repr(decision_fn)),
        "scores": scores,
        "metrics_by_config": configs_metrics,
    }


def paired_fold_tests(df: pd.DataFrame,
                      reference: str = "M1",
                      configs: Optional[list] = None,
                      metrics: tuple = ("NSE", "HitRatio"),
                      horizon: str = "48h",
                      strategy: Optional[str] = "expanding") -> Dict[str, dict]:
    """Paired across-fold test: Wilcoxon signed-rank of each configuration vs. ``reference``.

    Folds are paired by ``test_year``. Unlike the bootstrap confidence
    intervals, which are computed WITHIN each fold, this test compares
    configurations fold by fold and therefore supports claims that one
    configuration outperforms another.

    Args:
        df (pd.DataFrame): Consolidated results.
        reference (str): Reference configuration.
        configs (Optional[list]): Configurations to test (default: M1-M5 except the reference).
        metrics (tuple): Metrics to compare.
        horizon (str): Horizon to keep.
        strategy (Optional[str]): Strategy to keep (``None`` keeps all).

    Returns:
        Dict[str, dict]: Per configuration and metric: number of pairs, mean
        and standard deviation of the difference, Wilcoxon p-value (``None``
        with fewer than 3 valid pairs or all-zero differences, where the test
        is undefined), number of zero differences and whether p is exact.
    """
    from scipy.stats import wilcoxon

    if "horizon" in df.columns:
        df = df[df["horizon"] == horizon]
    if strategy is not None and "strategy" in df.columns:
        df = df[df["strategy"] == strategy]
    df = filter_published_variant(df)

    configs = configs or [c for c in M1_M5 if c != reference]
    ref = df[df["config"] == reference].set_index("test_year")
    out: Dict[str, dict] = {}

    for cfg in configs:
        cur = df[df["config"] == cfg].set_index("test_year")
        years = sorted(set(ref.index) & set(cur.index))
        out[cfg] = {"n_folds": len(years)}
        for m in metrics:
            if m not in df.columns or not years:
                out[cfg][m] = None
                continue
            pairs = pd.DataFrame({"ref": ref.loc[years, m], "cur": cur.loc[years, m]}).dropna()
            diff = (pairs["cur"] - pairs["ref"]).to_numpy()
            # wilcoxon is undefined with <3 pairs or all-zero differences
            # (degenerate ranks): report None instead of raising or returning
            # a meaningless p-value.
            p = None
            if diff.size >= 3 and np.any(diff != 0):
                p = float(wilcoxon(pairs["cur"], pairs["ref"], nan_policy="omit").pvalue)
            n_zero = int(np.sum(diff == 0))
            out[cfg][m] = {
                "n_pairs": int(diff.size),
                "mean_diff": float(np.mean(diff)) if diff.size else None,
                "std_diff": float(np.std(diff, ddof=1)) if diff.size > 1 else None,
                "wilcoxon_p": p,
                # With any exactly-zero difference scipy drops the exact
                # computation and uses the normal approximation, unreliable
                # with such a small n (7 folds): p must be reported as approximate.
                "n_zero_diffs": n_zero,
                "p_exact": p is not None and n_zero == 0,
            }
    return out


def run_model_selection(
    results_csv_path: str,
    out_json_path: Optional[str] = None,
    decision_fn: Callable[[Dict[str, dict]], Dict[str, float]] = lexicographic_constrained_score,
    horizon: str = "48h",
    exclude_holdout: bool = True,
) -> Dict:
    """Aggregate the results per configuration and select M*.

    M* is always selected on a single horizon (48h by default), never mixing
    24h/48h/72h in the same mean, and on pure cross-validation: the blind
    holdout is reported separately by :func:`build_final_report`, never used
    as selection evidence.

    Args:
        results_csv_path (str): Path to ``results_consolidated.csv``.
        out_json_path (Optional[str]): If given, the result is saved as JSON.
        decision_fn (Callable[[Dict[str, dict]], Dict[str, float]]): Decision function.
        horizon (str): Horizon to keep.
        exclude_holdout (bool): Exclude the blind-holdout rows.

    Returns:
        Dict: Output of :func:`select_best_model`.
    """
    df = pd.read_csv(results_csv_path)
    configs_metrics = aggregate_config_metrics(df, horizon=horizon, exclude_holdout=exclude_holdout)
    result = select_best_model(configs_metrics, decision_fn=decision_fn)

    if out_json_path:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def build_final_report(
    results_csv_path: str,
    out_json_path: Optional[str] = None,
    decision_fn: Callable[[Dict[str, dict]], Dict[str, float]] = lexicographic_constrained_score,
    horizon: str = "48h",
) -> Dict:
    """Join the selection of M* with its blind-holdout metrics in one report.

    Args:
        results_csv_path (str): Path to ``results_consolidated.csv``.
        out_json_path (Optional[str]): If given, the report is saved as JSON.
        decision_fn (Callable[[Dict[str, dict]], Dict[str, float]]): Decision function.
        horizon (str): Horizon to keep.

    Returns:
        Dict: Selection result plus ``holdout_metrics`` (the
        ``strategy="holdout_final"`` row of M*, or ``None``).
    """
    result = run_model_selection(results_csv_path, decision_fn=decision_fn, horizon=horizon)
    df = pd.read_csv(results_csv_path)
    holdout_mask = (df["config"] == result["selected"]) & (df["strategy"] == "holdout_final")
    if "horizon" in df.columns:
        holdout_mask &= df["horizon"] == horizon
    holdout_rows = df[holdout_mask]
    result["holdout_metrics"] = holdout_rows.iloc[0].to_dict() if not holdout_rows.empty else None

    if out_json_path:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=float)

    return result
