"""Reference baselines compared against the Seq2Seq + Random Forest pipeline.

All baselines accept 2-D ``(N, features)`` or 3-D ``(N, timesteps, features)``
inputs (3-D inputs are flattened to ``(N, timesteps * features)``) and return
predictions of shape ``(N, horizon)``.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import LinAlgError
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor, SGDRegressor


def _flatten_2d(X: ArrayLike) -> np.ndarray:
    """Return ``X`` as a float 2-D array, flattening 3-D windows.

    Args:
        X (ArrayLike): Array with 2 or 3 dimensions.

    Returns:
        np.ndarray: Array of shape ``(N, features)``.

    Raises:
        ValueError: If ``X`` does not have 2 or 3 dimensions.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim != 2:
        raise ValueError(f"Expected X with 2 or 3 dimensions, got ndim={X.ndim}.")
    return X


def _as_2d_targets(y: ArrayLike) -> np.ndarray:
    """Return targets as a float 2-D array ``(N, horizon)``.

    Args:
        y (ArrayLike): Targets with 1 or 2 dimensions.

    Returns:
        np.ndarray: Array of shape ``(N, horizon)``.

    Raises:
        ValueError: If ``y`` does not have 1 or 2 dimensions.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim != 2:
        raise ValueError(f"Expected y with 1 or 2 dimensions, got ndim={y.ndim}.")
    return y


class QuantileRegressionBaseline:
    """Linear quantile regression, one independent model per horizon step.

    If ``QuantileRegressor`` fails (e.g. an ill-conditioned optimization), the
    step falls back to ``SGDRegressor`` with an epsilon-insensitive loss.

    Args:
        quantile (float): Target quantile in ``(0, 1)``.
        alpha (float): L1 regularization strength of ``QuantileRegressor``
            (not the quantile).
        solver (str): ``QuantileRegressor`` solver.
        random_state (int): Seed of the ``SGDRegressor`` fallback.

    Raises:
        ValueError: If ``quantile`` is not in ``(0, 1)``.
    """

    def __init__(self, quantile: float = 0.5, alpha: float = 0.0, solver: str = "highs", random_state: int = 42) -> None:
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile must be in (0, 1).")
        self.quantile = quantile
        self.alpha = alpha
        self.solver = solver
        self.random_state = random_state
        self._models: list = []
        self.n_outputs_: int = 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "QuantileRegressionBaseline":
        """Fit one model per horizon step.

        Args:
            X (ArrayLike): Input windows, 2-D or 3-D.
            y (ArrayLike): Targets, 1-D or 2-D.

        Returns:
            QuantileRegressionBaseline: The fitted instance.

        Raises:
            ValueError: If ``X`` and ``y`` have different numbers of samples.
        """
        X2d = _flatten_2d(X)
        y2d = _as_2d_targets(y)
        if X2d.shape[0] != y2d.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self._models = []
        for h in range(y2d.shape[1]):
            try:
                model = QuantileRegressor(quantile=self.quantile, alpha=self.alpha, solver=self.solver)
                model.fit(X2d, y2d[:, h])
            except (LinAlgError, ValueError) as exc:
                warnings.warn(
                    f"QuantileRegressionBaseline: QuantileRegressor failed at horizon "
                    f"h={h} ({exc!r}); using SGDRegressor(epsilon_insensitive) as a "
                    f"fallback approximation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                model = SGDRegressor(loss="epsilon_insensitive", epsilon=0.0, random_state=self.random_state)
                model.fit(X2d, y2d[:, h])
            self._models.append(model)
        self.n_outputs_ = len(self._models)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict every horizon step.

        Args:
            X (ArrayLike): Input windows, 2-D or 3-D.

        Returns:
            np.ndarray: Predictions of shape ``(N, horizon)``.

        Raises:
            RuntimeError: If the baseline has not been fitted.
        """
        if not self._models:
            raise RuntimeError("QuantileRegressionBaseline has not been fitted (call fit() first).")
        X2d = _flatten_2d(X)
        preds = [m.predict(X2d) for m in self._models]
        return np.stack(preds, axis=1)


class GradientBoostingQuantileBaseline:
    """Gradient boosting quantile regression, one model per horizon step.

    Each step is a ``GradientBoostingRegressor(loss="quantile", alpha=quantile)``;
    in scikit-learn the quantile of this loss is passed through ``alpha``.

    Args:
        quantile (float): Target quantile in ``(0, 1)``.
        n_estimators (int): Number of boosting stages.
        max_depth (int): Maximum depth of each tree.
        learning_rate (float): Shrinkage.
        random_state (int): Random seed.

    Raises:
        ValueError: If ``quantile`` is not in ``(0, 1)``.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile must be in (0, 1).")
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._models: list = []
        self.n_outputs_: int = 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GradientBoostingQuantileBaseline":
        """Fit one model per horizon step.

        Args:
            X (ArrayLike): Input windows, 2-D or 3-D.
            y (ArrayLike): Targets, 1-D or 2-D.

        Returns:
            GradientBoostingQuantileBaseline: The fitted instance.

        Raises:
            ValueError: If ``X`` and ``y`` have different numbers of samples.
        """
        X2d = _flatten_2d(X)
        y2d = _as_2d_targets(y)
        if X2d.shape[0] != y2d.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self._models = []
        for h in range(y2d.shape[1]):
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=self.quantile,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
            model.fit(X2d, y2d[:, h])
            self._models.append(model)
        self.n_outputs_ = len(self._models)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict every horizon step.

        Args:
            X (ArrayLike): Input windows, 2-D or 3-D.

        Returns:
            np.ndarray: Predictions of shape ``(N, horizon)``.

        Raises:
            RuntimeError: If the baseline has not been fitted.
        """
        if not self._models:
            raise RuntimeError("GradientBoostingQuantileBaseline has not been fitted (call fit() first).")
        X2d = _flatten_2d(X)
        preds = [m.predict(X2d) for m in self._models]
        return np.stack(preds, axis=1)


class HeuristicEnsembleBaseline:
    """Static combination of several base models' predictions.

    ``X`` is a stack of per-model predictions (not raw input windows), with
    shape ``(N, n_predictors)`` (horizon 1) or ``(N, horizon, n_predictors)``.
    ``fit(X, y)`` learns per-model weights inversely proportional to each
    model's MAE against ``y`` (unless explicit ``weights`` are given). Without
    ``fit()``, :meth:`predict_weighted_mean` uses uniform weights.

    Args:
        weights (Optional[Sequence[float]]): Fixed per-model weights.
        mode (str): ``"weighted"`` or ``"simple"`` mean.

    Raises:
        ValueError: If ``mode`` is not ``"weighted"`` or ``"simple"``.
    """

    def __init__(self, weights: Optional[Sequence[float]] = None, mode: str = "weighted") -> None:
        if mode not in ("weighted", "simple"):
            raise ValueError("mode must be 'weighted' or 'simple'.")
        self.weights = np.asarray(weights, dtype=float) if weights is not None else None
        self.mode = mode
        self._fitted_weights: Optional[np.ndarray] = None

    @staticmethod
    def _prep(X: ArrayLike) -> np.ndarray:
        """Return predictions as a 3-D array ``(N, horizon, n_predictors)``.

        Args:
            X (ArrayLike): Stacked predictions, 2-D or 3-D.

        Returns:
            np.ndarray: 3-D float array.

        Raises:
            ValueError: If ``X`` does not have 2 or 3 dimensions.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[:, None, :]
        elif X.ndim != 3:
            raise ValueError(
                "HeuristicEnsembleBaseline expects X with shape (N, n_predictors) "
                "or (N, horizon, n_predictors)."
            )
        return X

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "HeuristicEnsembleBaseline":
        """Set the combination weights.

        Args:
            X (ArrayLike): Stacked predictions, 2-D or 3-D.
            y (Optional[ArrayLike]): Observed targets used for inverse-MAE
                weighting; if ``None`` and no fixed weights, weights are uniform.

        Returns:
            HeuristicEnsembleBaseline: The fitted instance.

        Raises:
            ValueError: If fixed weights do not match the number of predictors.
        """
        X3d = self._prep(X)
        n_predictors = X3d.shape[-1]

        if self.weights is not None:
            w = self.weights
            if w.shape[0] != n_predictors:
                raise ValueError("The number of weights does not match n_predictors.")
        elif y is not None:
            y2d = _as_2d_targets(y)
            errors = np.array([
                np.mean(np.abs(X3d[..., k] - y2d)) for k in range(n_predictors)
            ])
            inv = 1.0 / np.clip(errors, 1e-9, None)
            w = inv / inv.sum()
        else:
            w = np.full(n_predictors, 1.0 / n_predictors)

        self._fitted_weights = w
        return self

    def predict_simple_mean(self, X: ArrayLike) -> np.ndarray:
        """Unweighted mean across predictors.

        Args:
            X (ArrayLike): Stacked predictions, 2-D or 3-D.

        Returns:
            np.ndarray: Predictions of shape ``(N, horizon)``.
        """
        X3d = self._prep(X)
        return X3d.mean(axis=-1)

    def predict_weighted_mean(self, X: ArrayLike) -> np.ndarray:
        """Weighted mean across predictors.

        Args:
            X (ArrayLike): Stacked predictions, 2-D or 3-D.

        Returns:
            np.ndarray: Predictions of shape ``(N, horizon)``.

        Raises:
            ValueError: If the fitted weights do not match the predictors in ``X``.
        """
        X3d = self._prep(X)
        if self._fitted_weights is not None:
            w = self._fitted_weights
        else:
            w = np.full(X3d.shape[-1], 1.0 / X3d.shape[-1])
        if w.shape[0] != X3d.shape[-1]:
            raise ValueError("The number of fitted weights does not match n_predictors of X.")
        return np.tensordot(X3d, w, axes=([2], [0]))

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict with the configured ``mode``.

        Args:
            X (ArrayLike): Stacked predictions, 2-D or 3-D.

        Returns:
            np.ndarray: Predictions of shape ``(N, horizon)``.
        """
        return self.predict_simple_mean(X) if self.mode == "simple" else self.predict_weighted_mean(X)
