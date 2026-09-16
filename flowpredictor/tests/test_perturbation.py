"""Tests for evaluation/perturbation.py.

Uniform ±10% and Gaussian ±5% noise on the rainfall columns, and the Monte
Carlo test with 50 replicas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resultados.prueba63.src.evaluation.perturbation import (
    DEFAULT_RAIN_COLUMNS,
    perturb_rainfall_uniform,
    perturb_rainfall_gaussian,
    run_perturbation_test,
)


def _make_df(n=200, rain_value=10.0):
    return pd.DataFrame({
        "Fecha": pd.date_range("2020-01-01", periods=n),
        "Qe": np.linspace(1, 100, n),
        "l/m2_arzua": np.full(n, rain_value),
        "l/m2_serradofaro": np.full(n, rain_value),
        "l/m2_melide": np.full(n, rain_value),
        "l/m2_olveda": np.full(n, rain_value),
        "pred_l/m2": np.full(n, rain_value),
        "pred_l/m2_2d": np.full(n, rain_value),
        "pred_l/m2_3d": np.full(n, rain_value),
    })


# --------------------------------------------------------------------------- #
# perturb_rainfall_uniform / perturb_rainfall_gaussian
# --------------------------------------------------------------------------- #

def test_uniform_perturbation_stays_within_plus_minus_10_percent():
    df = _make_df(rain_value=10.0)
    out = perturb_rainfall_uniform(df, frac=0.10, seed=1)
    for col in DEFAULT_RAIN_COLUMNS:
        assert (out[col] >= 9.0 - 1e-9).all()
        assert (out[col] <= 11.0 + 1e-9).all()


def test_uniform_perturbation_is_deterministic_given_seed():
    df = _make_df()
    out1 = perturb_rainfall_uniform(df, frac=0.10, seed=5)
    out2 = perturb_rainfall_uniform(df, frac=0.10, seed=5)
    pd.testing.assert_frame_equal(out1, out2)


def test_uniform_perturbation_changes_values_with_high_probability():
    df = _make_df(n=500, rain_value=10.0)
    out = perturb_rainfall_uniform(df, frac=0.10, seed=2)
    assert not np.allclose(out["l/m2_arzua"].to_numpy(), df["l/m2_arzua"].to_numpy())


def test_uniform_perturbation_only_touches_requested_columns():
    df = _make_df()
    out = perturb_rainfall_uniform(df, rain_cols=["l/m2_arzua"], frac=0.10, seed=3)
    np.testing.assert_array_equal(out["l/m2_serradofaro"].to_numpy(), df["l/m2_serradofaro"].to_numpy())
    np.testing.assert_array_equal(out["Qe"].to_numpy(), df["Qe"].to_numpy())


def test_uniform_perturbation_never_negative():
    df = _make_df(rain_value=0.05)  # small value, close to 0
    out = perturb_rainfall_uniform(df, frac=0.10, seed=4)
    for col in DEFAULT_RAIN_COLUMNS:
        assert (out[col] >= 0.0).all()


def test_gaussian_perturbation_std_matches_sigma_frac_approximately():
    df = _make_df(n=20000, rain_value=10.0)  # large sample to estimate sigma accurately
    out = perturb_rainfall_gaussian(df, rain_cols=["l/m2_arzua"], sigma_frac=0.05, seed=6)
    empirical_std = out["l/m2_arzua"].std()
    expected_std = 10.0 * 0.05
    assert empirical_std == pytest.approx(expected_std, rel=0.1)


def test_gaussian_perturbation_is_deterministic_given_seed():
    df = _make_df()
    out1 = perturb_rainfall_gaussian(df, sigma_frac=0.05, seed=8)
    out2 = perturb_rainfall_gaussian(df, sigma_frac=0.05, seed=8)
    pd.testing.assert_frame_equal(out1, out2)


def test_gaussian_perturbation_never_negative():
    df = _make_df(rain_value=0.02)
    out = perturb_rainfall_gaussian(df, sigma_frac=0.05, seed=9)
    for col in DEFAULT_RAIN_COLUMNS:
        assert (out[col] >= 0.0).all()


# --------------------------------------------------------------------------- #
# run_perturbation_test: 50 Monte Carlo replicas, both noise types
# --------------------------------------------------------------------------- #

def _sum_rain_predict_fn(df: pd.DataFrame) -> np.ndarray:
    return df[DEFAULT_RAIN_COLUMNS].sum(axis=1).to_numpy()


def test_run_perturbation_test_default_50_replicas_both_noise_types():
    df = _make_df(n=30)
    out = run_perturbation_test(_sum_rain_predict_fn, df, n_replicas=50, seed=42)

    assert set(out.keys()) == {"uniform", "gaussian"}
    for noise_type in ("uniform", "gaussian"):
        result = out[noise_type]
        assert result["predictions"].shape == (50, 30)
        assert result["mean"].shape == (30,)
        assert result["std"].shape == (30,)
        assert result["baseline"].shape == (30,)
        assert result["n_replicas"] == 50


def test_run_perturbation_test_uses_requested_fracs():
    df = _make_df(n=10)
    out = run_perturbation_test(
        _sum_rain_predict_fn, df, n_replicas=5, uniform_frac=0.10, gaussian_frac=0.05, seed=1,
    )
    assert out["uniform"]["frac"] == 0.10
    assert out["gaussian"]["frac"] == 0.05


def test_run_perturbation_test_std_is_nonzero_for_noisy_predict_fn():
    """The prediction depends linearly on the perturbed rainfall, so with real
    injected noise the spread across replicas must be > 0 at every point."""
    df = _make_df(n=15, rain_value=10.0)
    out = run_perturbation_test(_sum_rain_predict_fn, df, n_replicas=50, seed=3)
    assert (out["uniform"]["std"] > 0).all()
    assert (out["gaussian"]["std"] > 0).all()


def test_run_perturbation_test_constant_predict_fn_has_zero_std():
    """If predict_fn ignores its input, the spread across replicas must be
    exactly 0 (there is no randomness outside the perturbation)."""
    df = _make_df(n=8)

    def constant_predict_fn(_df):
        return np.full(8, 42.0)

    out = run_perturbation_test(constant_predict_fn, df, n_replicas=10, seed=1)
    assert np.allclose(out["uniform"]["std"], 0.0)
    assert np.allclose(out["gaussian"]["std"], 0.0)
    assert np.allclose(out["uniform"]["mean"], 42.0)


def test_run_perturbation_test_is_deterministic_given_seed():
    df = _make_df(n=12)
    out1 = run_perturbation_test(_sum_rain_predict_fn, df, n_replicas=20, seed=99)
    out2 = run_perturbation_test(_sum_rain_predict_fn, df, n_replicas=20, seed=99)

    np.testing.assert_allclose(out1["uniform"]["predictions"], out2["uniform"]["predictions"])
    np.testing.assert_allclose(out1["gaussian"]["predictions"], out2["gaussian"]["predictions"])


# --------------------------------------------------------------------------- #
# delta_hit_ratio/delta_farate (only when y_true is given)
# --------------------------------------------------------------------------- #

def test_run_perturbation_test_without_y_true_has_no_delta_keys():
    """Default behavior (y_true=None): no delta keys."""
    df = _make_df(n=10)
    out = run_perturbation_test(_sum_rain_predict_fn, df, n_replicas=5, seed=1)
    assert "delta_hit_ratio" not in out["uniform"]
    assert "delta_farate" not in out["uniform"]


def test_run_perturbation_test_constant_predict_fn_has_zero_delta_hit_ratio():
    """predict_fn ignores the noise -> the replica mean equals the baseline ->
    HitRatio/FARate do not change -> delta is exactly 0."""
    df = _make_df(n=8)
    y_true = np.full(8, 42.0)

    def constant_predict_fn(_df):
        return np.full(8, 42.0)

    out = run_perturbation_test(
        constant_predict_fn, df, n_replicas=10, seed=1, y_true=y_true, threshold_p90=40.0,
    )
    assert out["uniform"]["delta_hit_ratio"] == pytest.approx(0.0)
    assert out["uniform"]["delta_farate"] == pytest.approx(0.0)
    assert out["gaussian"]["delta_hit_ratio"] == pytest.approx(0.0)


def test_run_perturbation_test_delta_hit_ratio_reflects_degradation_under_noise():
    """predict_fn depends linearly on total rainfall: perturbing it changes the
    replica mean relative to the baseline and can therefore change
    HitRatio/FARate; the delta keys must be present and numeric."""
    df = _make_df(n=30, rain_value=10.0)
    y_true = df[DEFAULT_RAIN_COLUMNS].sum(axis=1).to_numpy()  # == exact baseline (predict_fn = rainfall sum)
    threshold = float(np.percentile(y_true, 90))

    out = run_perturbation_test(
        _sum_rain_predict_fn, df, n_replicas=50, seed=3, y_true=y_true, threshold_p90=threshold,
    )
    for noise_type in ("uniform", "gaussian"):
        assert "delta_hit_ratio" in out[noise_type]
        assert "delta_farate" in out[noise_type]
        assert isinstance(out[noise_type]["delta_hit_ratio"], float)
