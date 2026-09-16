"""Random Forest wrappers used as the M5 meta-learner (branch selector)."""
from __future__ import annotations

import pickle
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RFClassifierWrapper:
    """Random Forest classifier over flattened context and future windows.

    Args:
        num_models (int): Number of candidate models (classes).
        input_context_shape (Tuple[int, ...]): Per-sample encoder window shape.
        input_future_shape (Tuple[int, ...]): Per-sample decoder window shape.
        seed (Optional[int]): Random state; defaults to 42.
    """

    def __init__(self, num_models: int, input_context_shape: Tuple[int, ...], input_future_shape: Tuple[int, ...], seed: Optional[int] = None) -> None:
        self.num_models = num_models
        self.input_context_shape = input_context_shape
        self.input_future_shape = input_future_shape
        self.seed = seed if seed is not None else 42
        # Dummy fit with one sample per class so that every class is known.
        self.model = RandomForestClassifier(n_estimators=800, max_features="sqrt", class_weight="balanced_subsample", random_state=self.seed,)
        dummy_context = np.zeros((1,) + self.input_context_shape)
        dummy_future = np.zeros((1,) + self.input_future_shape)
        x_context_flat = dummy_context.reshape(1, -1)
        x_future_flat = dummy_future.reshape(1, -1)
        X_dummy = np.concatenate([x_context_flat, x_future_flat], axis=1)
        dummy_y = np.arange(self.num_models)
        X_dummy = np.tile(X_dummy, (self.num_models, 1))
        self.model.fit(X_dummy, dummy_y)

    def fit(self, inputs: Sequence[np.ndarray], y: np.ndarray, **kwargs: Any) -> None:
        """Refit the classifier on real data.

        Args:
            inputs (Sequence[np.ndarray]): ``(x_context, x_future)`` windows.
            y (np.ndarray): Class indices, or one-hot labels.
            **kwargs (Any): Extra ``RandomForestClassifier`` arguments; Keras
                training arguments (``epochs``, ``batch_size``, ...) are dropped.
        """
        # Drop arguments that are not valid for RandomForestClassifier.
        for invalid in ['epochs', 'batch_size', 'validation_split', 'max_epochs', 'max_epochs_clasificador']:
            kwargs.pop(invalid, None)

        x_context, x_future = inputs
        x_context_flat = x_context.reshape(x_context.shape[0], -1)
        x_future_flat = x_future.reshape(x_future.shape[0], -1)
        X = np.concatenate([x_context_flat, x_future_flat], axis=1)

        # Re-create the model with the valid arguments only.
        self.model = RandomForestClassifier(
            n_estimators=kwargs.pop("n_estimators", 800),
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=self.seed,
            **kwargs  # optional extras such as max_depth, min_samples_leaf
        )

        # Convert one-hot labels to class indices when needed.
        if y.ndim > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        self.model.fit(X, y)

    def predict(self, inputs: Sequence[np.ndarray]) -> np.ndarray:
        """Predict class probabilities for every candidate model.

        Args:
            inputs (Sequence[np.ndarray]): ``(x_context, x_future)`` windows.

        Returns:
            np.ndarray: Probabilities of shape ``(n_samples, num_models)``.
        """
        x_context, x_future = inputs
        x_context_flat = x_context.reshape(x_context.shape[0], -1)
        x_future_flat = x_future.reshape(x_future.shape[0], -1)
        X = np.concatenate([x_context_flat, x_future_flat], axis=1)
        probs = self.model.predict_proba(X)

        # Ensure the output always has shape (n_samples, num_models).
        full_probs = np.zeros((X.shape[0], self.num_models))
        for idx, cls in enumerate(self.model.classes_):
            full_probs[:, cls] = probs[:, idx]
        return full_probs

    def save(self, filepath: str) -> None:
        """Pickle the wrapper to ``filepath``.

        Args:
            filepath (str): Destination path.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> "RFClassifierWrapper":
        """Load a pickled wrapper.

        Args:
            filepath (str): Source path.

        Returns:
            RFClassifierWrapper: The unpickled wrapper.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class RFRegressorWrapper:
    """Random Forest regressor that outputs a continuous branch index.

    The M5 selector is trained on the flattened encoder window, decoder window
    and the candidate models' predictions; its continuous output is later
    rounded to the nearest branch.

    Args:
        num_models (int): Number of candidate models (kept for API symmetry).
        input_context_shape (Tuple[int, ...]): Per-sample encoder window shape.
        input_future_shape (Tuple[int, ...]): Per-sample decoder window shape.
        seed (Optional[int]): Random state; defaults to 42.
    """

    def __init__(self, num_models: int, input_context_shape: Tuple[int, ...], input_future_shape: Tuple[int, ...], seed: Optional[int] = None) -> None:
        self.input_context_shape = input_context_shape
        self.input_future_shape = input_future_shape
        self.seed = seed if seed is not None else 42
        self.model = RandomForestRegressor(
            n_estimators=800,
            max_features="sqrt",
            random_state=self.seed,
        )

    def fit(self, inputs: Sequence[np.ndarray], y: np.ndarray, **kwargs: Any) -> None:
        """Fit the regressor.

        Args:
            inputs (Sequence[np.ndarray]): ``(x_context, x_future, base_preds)``,
                where ``base_preds`` holds the candidate models' predictions.
            y (np.ndarray): Target branch index per sample.
            **kwargs (Any): Extra ``RandomForestRegressor`` arguments; Keras
                training arguments are dropped.
        """
        for invalid in ['epochs', 'batch_size', 'validation_split', 'max_epochs', 'max_epochs_clasificador']:
            kwargs.pop(invalid, None)

        # Three inputs: context, future and the candidate models' predictions.
        x_context, x_future, base_preds = inputs
        X = np.concatenate([
            x_context.reshape(x_context.shape[0], -1),
            x_future.reshape(x_future.shape[0], -1),
            base_preds  # candidate predictions used as input features
        ], axis=1)

        self.model = RandomForestRegressor(
            n_estimators=kwargs.pop("n_estimators", 800),
            max_features="sqrt",
            random_state=self.seed,
            **kwargs
        )
        self.model.fit(X, y)

    def predict(self, inputs: Sequence[np.ndarray]) -> np.ndarray:
        """Predict the continuous branch index.

        Args:
            inputs (Sequence[np.ndarray]): ``(x_context, x_future, base_preds)``.

        Returns:
            np.ndarray: Continuous branch index per sample.
        """
        x_context, x_future, base_preds = inputs
        X = np.concatenate([
            x_context.reshape(x_context.shape[0], -1),
            x_future.reshape(x_future.shape[0], -1),
            base_preds
        ], axis=1)
        return self.model.predict(X)

    def save(self, filepath: str) -> None:
        """Pickle the wrapper to ``filepath``.

        Args:
            filepath (str): Destination path.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> "RFRegressorWrapper":
        """Load a pickled wrapper.

        Args:
            filepath (str): Source path.

        Returns:
            RFRegressorWrapper: The unpickled wrapper.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def build_random_forest_classifier(input_context_shape: Tuple[int, ...], input_future_shape: Tuple[int, ...], num_models: int, iteration_params: Dict[str, Any]) -> RFClassifierWrapper:
    """Create an :class:`RFClassifierWrapper`.

    Args:
        input_context_shape (Tuple[int, ...]): Per-sample encoder window shape.
        input_future_shape (Tuple[int, ...]): Per-sample decoder window shape.
        num_models (int): Number of candidate models.
        iteration_params (Dict[str, Any]): Iteration hyperparameters; only
            ``seed`` is read here, the rest is consumed later by ``fit``.

    Returns:
        RFClassifierWrapper: The untrained wrapper.
    """
    seed = iteration_params.get("seed", 42)
    return RFClassifierWrapper(num_models, input_context_shape, input_future_shape, seed=seed)


def build_random_forest_regressor(input_context_shape: Tuple[int, ...], input_future_shape: Tuple[int, ...], num_models: int, iteration_params: Dict[str, Any]) -> RFRegressorWrapper:
    """Create an :class:`RFRegressorWrapper`.

    Args:
        input_context_shape (Tuple[int, ...]): Per-sample encoder window shape.
        input_future_shape (Tuple[int, ...]): Per-sample decoder window shape.
        num_models (int): Number of candidate models.
        iteration_params (Dict[str, Any]): Iteration hyperparameters; only
            ``seed`` is read here, the rest is consumed later by ``fit``.

    Returns:
        RFRegressorWrapper: The untrained wrapper.
    """
    seed = iteration_params.get("seed", 42)
    return RFRegressorWrapper(num_models, input_context_shape, input_future_shape, seed=seed)
