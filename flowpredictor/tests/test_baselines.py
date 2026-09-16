"""Shape and fit/predict tests for resultados/prueba63/src/models/baselines.py."""
from __future__ import annotations

import numpy as np
import pytest

from resultados.prueba63.src.models.baselines import (
    QuantileRegressionBaseline,
    GradientBoostingQuantileBaseline,
    HeuristicEnsembleBaseline,
)


def _linear_dataset(n=60, n_features=4, horizon=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, n_features))
    W = rng.uniform(0.5, 1.5, size=(n_features, horizon))
    y = X @ W + rng.normal(0, 0.01, size=(n, horizon))
    return X, y


# --------------------------------------------------------------------------- #
# QuantileRegressionBaseline
# --------------------------------------------------------------------------- #

def test_quantile_regression_baseline_shapes_2d_input():
    X, y = _linear_dataset(n=50, n_features=3, horizon=2)
    model = QuantileRegressionBaseline(quantile=0.5)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (50, 2)
    assert model.n_outputs_ == 2


def test_quantile_regression_baseline_shapes_3d_input_is_flattened():
    n, T, F, horizon = 40, 5, 3, 1
    rng = np.random.default_rng(1)
    X3d = rng.uniform(-1, 1, size=(n, T, F))
    y = rng.uniform(0, 1, size=(n,))  # 1D target -> horizon=1

    model = QuantileRegressionBaseline(quantile=0.5)
    model.fit(X3d, y)
    preds = model.predict(X3d)

    assert preds.shape == (n, 1)


def test_quantile_regression_baseline_fits_linear_relationship_reasonably():
    X, y = _linear_dataset(n=200, n_features=3, horizon=1, seed=2)
    model = QuantileRegressionBaseline(quantile=0.5, alpha=0.0)
    model.fit(X, y)
    preds = model.predict(X)

    mae = np.mean(np.abs(preds - y))
    assert mae < 0.5  # injected noise has sigma=0.01, the fit must be good


def test_quantile_regression_baseline_predict_before_fit_raises():
    model = QuantileRegressionBaseline()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 3)))


def test_quantile_regression_baseline_invalid_quantile_raises():
    with pytest.raises(ValueError):
        QuantileRegressionBaseline(quantile=1.5)


def test_quantile_regression_baseline_higher_quantile_shifts_predictions_up():
    X, y = _linear_dataset(n=200, n_features=3, horizon=1, seed=3)
    low = QuantileRegressionBaseline(quantile=0.1).fit(X, y).predict(X)
    high = QuantileRegressionBaseline(quantile=0.9).fit(X, y).predict(X)

    assert np.mean(high) > np.mean(low)


# --------------------------------------------------------------------------- #
# GradientBoostingQuantileBaseline
# --------------------------------------------------------------------------- #

def test_gb_quantile_baseline_shapes_2d_input():
    X, y = _linear_dataset(n=60, n_features=4, horizon=3)
    model = GradientBoostingQuantileBaseline(quantile=0.5, n_estimators=20)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (60, 3)
    assert model.n_outputs_ == 3


def test_gb_quantile_baseline_shapes_3d_input_is_flattened():
    n, T, F = 30, 4, 2
    rng = np.random.default_rng(4)
    X3d = rng.uniform(-1, 1, size=(n, T, F))
    y = rng.uniform(0, 1, size=(n, 2))

    model = GradientBoostingQuantileBaseline(quantile=0.5, n_estimators=15)
    model.fit(X3d, y)
    preds = model.predict(X3d)

    assert preds.shape == (n, 2)


def test_gb_quantile_baseline_predict_before_fit_raises():
    model = GradientBoostingQuantileBaseline()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 3)))


def test_gb_quantile_baseline_invalid_quantile_raises():
    with pytest.raises(ValueError):
        GradientBoostingQuantileBaseline(quantile=0.0)


# --------------------------------------------------------------------------- #
# HeuristicEnsembleBaseline
# --------------------------------------------------------------------------- #

def test_heuristic_ensemble_simple_mean_matches_manual_average():
    rng = np.random.default_rng(5)
    N, horizon, n_predictors = 10, 3, 4
    X = rng.uniform(0, 10, size=(N, horizon, n_predictors))

    model = HeuristicEnsembleBaseline(mode="simple")
    preds = model.predict(X)

    assert preds.shape == (N, horizon)
    np.testing.assert_allclose(preds, X.mean(axis=-1))


def test_heuristic_ensemble_weighted_mean_with_fixed_weights():
    N, horizon, n_predictors = 5, 2, 3
    X = np.ones((N, horizon, n_predictors))
    X[..., 0] = 0.0
    X[..., 1] = 10.0
    X[..., 2] = 20.0
    weights = [0.5, 0.25, 0.25]

    model = HeuristicEnsembleBaseline(weights=weights, mode="weighted")
    model.fit(X)
    preds = model.predict(X)

    expected = 0.5 * 0.0 + 0.25 * 10.0 + 0.25 * 20.0
    assert preds.shape == (N, horizon)
    np.testing.assert_allclose(preds, np.full((N, horizon), expected))


def test_heuristic_ensemble_fit_learns_inverse_error_weights():
    N, horizon = 100, 1
    rng = np.random.default_rng(6)
    y_true = rng.uniform(0, 10, size=(N, horizon))
    good_pred = y_true + rng.normal(0, 0.1, size=(N, horizon))   # low error
    bad_pred = y_true + rng.normal(0, 5.0, size=(N, horizon))    # high error
    X = np.stack([good_pred[..., 0], bad_pred[..., 0]], axis=-1)  # (N, n_predictors)

    model = HeuristicEnsembleBaseline(mode="weighted")
    model.fit(X, y_true)

    assert model._fitted_weights.shape == (2,)
    assert model._fitted_weights.sum() == pytest.approx(1.0)
    assert model._fitted_weights[0] > model._fitted_weights[1]  # the "good" model gets a larger weight

    preds = model.predict(X)
    assert preds.shape == (N, horizon)


def test_heuristic_ensemble_2d_input_treated_as_single_horizon():
    N, n_predictors = 8, 3
    X = np.arange(N * n_predictors, dtype=float).reshape(N, n_predictors)

    model = HeuristicEnsembleBaseline(mode="simple")
    preds = model.predict(X)

    assert preds.shape == (N, 1)
    np.testing.assert_allclose(preds[:, 0], X.mean(axis=-1))


def test_heuristic_ensemble_weight_length_mismatch_raises():
    X = np.ones((5, 2, 3))
    model = HeuristicEnsembleBaseline(weights=[1.0, 1.0])  # only 2 weights for 3 predictors
    with pytest.raises(ValueError):
        model.fit(X)


def test_heuristic_ensemble_invalid_mode_raises():
    with pytest.raises(ValueError):
        HeuristicEnsembleBaseline(mode="bogus")


# --------------------------------------------------------------------------- #
# tau=0.90 variants (the registry must instantiate the stated quantile, not the median)
# --------------------------------------------------------------------------- #

def test_gradient_boosting_tau90_maps_quantile_to_sklearn_alpha():
    """GradientBoostingRegressor(loss='quantile') takes the quantile as `alpha`,
    not as a `quantile` parameter: if the wrapper stopped translating it, the
    model would silently fall back to the median."""
    model = GradientBoostingQuantileBaseline(quantile=0.9, n_estimators=20)
    X, y = _linear_dataset(n=60, n_features=3, horizon=2)
    model.fit(X, y)

    assert model.n_outputs_ == 2
    for sub in model._models:
        assert sub.loss == "quantile"
        assert sub.alpha == pytest.approx(0.9)

    preds = model.predict(X)
    assert preds.shape == (60, 2)
    assert np.isfinite(preds).all()


def test_quantile_regression_tau90_uses_quantile_not_l1_alpha():
    """In QuantileRegressor the quantile is `quantile` and `alpha` is the L1
    regularization: confusing them would set regularization 0.9 and leave the
    prediction at the median."""
    model = QuantileRegressionBaseline(quantile=0.9)
    assert model.quantile == pytest.approx(0.9)
    assert model.alpha == pytest.approx(0.0)

    X, y = _linear_dataset(n=60, n_features=3, horizon=2)
    model.fit(X, y)
    preds_90 = model.predict(X)

    median = QuantileRegressionBaseline(quantile=0.5)
    median.fit(X, y)
    preds_50 = median.predict(X)

    assert preds_90.shape == preds_50.shape == (60, 2)
    # tau=0.90 must lie above the median in aggregate.
    assert preds_90.mean() > preds_50.mean()


def test_registry_exposes_both_tau_variants_without_renaming_published_ones():
    """The published names (tau=0.50) are the 'config' key of consolidated
    rows: renaming them would break the continuity of the CSV and the tables."""
    from resultados.prueba63.src.main_pipeline import BASELINE_REGISTRY, SKLEARN_BASELINES

    assert "quantile_regression" in BASELINE_REGISTRY
    assert "gradient_boosting_quantile" in BASELINE_REGISTRY
    assert BASELINE_REGISTRY["quantile_regression"]().quantile == pytest.approx(0.5)
    assert BASELINE_REGISTRY["gradient_boosting_quantile"]().quantile == pytest.approx(0.5)
    assert BASELINE_REGISTRY["quantile_regression_tau90"]().quantile == pytest.approx(0.9)
    assert BASELINE_REGISTRY["gradient_boosting_tau90"]().quantile == pytest.approx(0.9)
    assert set(SKLEARN_BASELINES) == {
        "quantile_regression", "gradient_boosting_quantile",
        "quantile_regression_tau90", "gradient_boosting_tau90",
    }
