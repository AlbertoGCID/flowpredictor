"""Rainfall-perturbation test.

Injects noise into the observed and forecast rainfall columns and repeats
inference N times (Monte Carlo) to measure how much the inflow prediction
changes under small input perturbations -- a robustness/sensitivity check,
not an accuracy check.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import compute_extreme_metrics

DEFAULT_RAIN_COLUMNS: List[str] = [
    "l/m2_arzua", "l/m2_serradofaro", "l/m2_melide", "l/m2_olveda",
    "pred_l/m2", "pred_l/m2_2d", "pred_l/m2_3d",
]


def perturb_rainfall_uniform(
    df: pd.DataFrame,
    rain_cols: Sequence[str] = DEFAULT_RAIN_COLUMNS,
    frac: float = 0.10,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Multiplicative uniform noise ±frac (±10% by default) on the rainfall columns.

    Args:
        df (pd.DataFrame): Input data.
        rain_cols (Sequence[str]): Columns to perturb (missing ones are ignored).
        frac (float): Noise amplitude.
        seed (Optional[int]): Random seed.

    Returns:
        pd.DataFrame: Perturbed copy (rainfall clipped at 0).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    cols = [c for c in rain_cols if c in out.columns]
    for c in cols:
        noise = rng.uniform(-frac, frac, size=len(out))
        out[c] = out[c].to_numpy(dtype=float) * (1.0 + noise)
        out[c] = out[c].clip(lower=0.0)  # rainfall cannot be negative
    return out


def perturb_rainfall_gaussian(
    df: pd.DataFrame,
    rain_cols: Sequence[str] = DEFAULT_RAIN_COLUMNS,
    sigma_frac: float = 0.05,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Multiplicative Gaussian noise ~N(0, sigma_frac) (sigma=5% by default) on the rainfall columns.

    Args:
        df (pd.DataFrame): Input data.
        rain_cols (Sequence[str]): Columns to perturb (missing ones are ignored).
        sigma_frac (float): Noise standard deviation.
        seed (Optional[int]): Random seed.

    Returns:
        pd.DataFrame: Perturbed copy (rainfall clipped at 0).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    cols = [c for c in rain_cols if c in out.columns]
    for c in cols:
        noise = rng.normal(0.0, sigma_frac, size=len(out))
        out[c] = out[c].to_numpy(dtype=float) * (1.0 + noise)
        out[c] = out[c].clip(lower=0.0)
    return out


def run_perturbation_test(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    df: pd.DataFrame,
    rain_cols: Sequence[str] = DEFAULT_RAIN_COLUMNS,
    n_replicas: int = 50,
    uniform_frac: float = 0.10,
    gaussian_frac: float = 0.05,
    seed: int = 42,
    y_true: Optional[np.ndarray] = None,
    threshold_p90: Optional[float] = None,
) -> Dict[str, Dict]:
    """Monte Carlo rainfall-perturbation test.

    Runs ``n_replicas`` replicas of each noise type (uniform ±10%, Gaussian
    sigma 5%) on ``df``, calling ``predict_fn(perturbed_df)`` for each, and
    summarizes the spread of the predictions. ``predict_fn`` is model-agnostic:
    it may wrap the Seq2Seq + RF pipeline or any baseline, as long as it accepts
    a DataFrame with the columns of ``df`` and returns an ``(N,)`` or
    ``(N, horizon)`` array.

    Args:
        predict_fn (Callable[[pd.DataFrame], np.ndarray]): Inference function.
        df (pd.DataFrame): Unperturbed input data.
        rain_cols (Sequence[str]): Columns to perturb.
        n_replicas (int): Replicas per noise type.
        uniform_frac (float): Uniform noise amplitude.
        gaussian_frac (float): Gaussian noise standard deviation.
        seed (int): Master seed of the replica seeds.
        y_true (Optional[np.ndarray]): Observed values, to compute metric deltas.
        threshold_p90 (Optional[float]): Extreme-event threshold fixed on TRAIN.

    Returns:
        Dict[str, Dict]: Per noise type: ``predictions`` (n_replicas, ...),
        point-wise ``mean`` and ``std`` across replicas, unperturbed
        ``baseline``, ``n_replicas``, ``frac`` and, if ``y_true`` is given,
        ``delta_hit_ratio``/``delta_farate`` (metrics of the replica mean minus
        those of the unperturbed baseline).
    """
    rng = np.random.default_rng(seed)
    baseline_pred = np.asarray(predict_fn(df))
    baseline_extreme = (
        compute_extreme_metrics(y_true, baseline_pred, threshold_p90) if y_true is not None else None
    )

    out: Dict[str, Dict] = {}
    for noise_type, perturb_fn, frac in (
        ("uniform", perturb_rainfall_uniform, uniform_frac),
        ("gaussian", perturb_rainfall_gaussian, gaussian_frac),
    ):
        replicas = []
        for _ in range(n_replicas):
            rep_seed = int(rng.integers(0, 2**32 - 1))
            df_pert = perturb_fn(df, rain_cols=rain_cols, seed=rep_seed, **(
                {"frac": frac} if noise_type == "uniform" else {"sigma_frac": frac}
            ))
            replicas.append(np.asarray(predict_fn(df_pert)))

        stacked = np.stack(replicas, axis=0)
        mean_pred = stacked.mean(axis=0)
        result = {
            "predictions": stacked,
            "mean": mean_pred,
            "std": stacked.std(axis=0),
            "baseline": baseline_pred,
            "n_replicas": n_replicas,
            "frac": frac,
        }
        if y_true is not None:
            perturbed_extreme = compute_extreme_metrics(y_true, mean_pred, threshold_p90)
            result["delta_hit_ratio"] = perturbed_extreme["HitRatio"] - baseline_extreme["HitRatio"]
            result["delta_farate"] = perturbed_extreme["FARate"] - baseline_extreme["FARate"]
        out[noise_type] = result

    return out
