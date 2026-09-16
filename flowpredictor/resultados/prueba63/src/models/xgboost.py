"""XGBoost classifier wrapper (alternative meta-learner, not used by M1-M5)."""
from __future__ import annotations

import pickle
import random
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import xgboost as xgb

from ...log.log_config import get_logger

logger = get_logger()


class XGBClassifierWrapper:
    """XGBoost multi-class classifier over flattened context and future windows.

    Args:
        num_models (int): Number of candidate models (classes).
        input_context_shape (Tuple[int, ...]): Per-sample encoder window shape.
        input_future_shape (Tuple[int, ...]): Per-sample decoder window shape.
        iteration_params (Dict[str, Any]): Iteration hyperparameters; ``seed``
            seeds NumPy and Python's ``random``.
    """

    def __init__(self, num_models: int, input_context_shape: Tuple[int, ...], input_future_shape: Tuple[int, ...], iteration_params: Dict[str, Any]) -> None:
        self.num_models = num_models
        self.input_context_shape = input_context_shape
        self.input_future_shape = input_future_shape
        seed = iteration_params.get("seed", 42)
        np.random.seed(seed)
        random.seed(seed)

        # Dummy fit so that self.model is never None.
        self.model = xgb.XGBClassifier(objective="multi:softmax", num_class=self.num_models)
        dummy_context = np.zeros((1,) + self.input_context_shape)
        dummy_future = np.zeros((1,) + self.input_future_shape)
        x_context_flat = dummy_context.reshape(1, -1)
        x_future_flat = dummy_future.reshape(1, -1)
        X_dummy = np.concatenate([x_context_flat, x_future_flat], axis=1)
        dummy_y = np.zeros(1, dtype=int)
        self.model.fit(X_dummy, dummy_y)

    def fit(self, inputs: Sequence[np.ndarray], y: np.ndarray, **kwargs: Any) -> None:
        """Refit the classifier on real data.

        Args:
            inputs (Sequence[np.ndarray]): ``(x_context, x_future)`` windows.
            y (np.ndarray): One-hot labels.
            **kwargs (Any): Extra ``XGBClassifier`` arguments.
        """
        x_context, x_future = inputs
        x_context_flat = x_context.reshape(x_context.shape[0], -1)
        x_future_flat = x_future.reshape(x_future.shape[0], -1)
        X = np.concatenate([x_context_flat, x_future_flat], axis=1)
        self.model = xgb.XGBClassifier(objective="multi:softmax", num_class=self.num_models, **kwargs)
        self.model.fit(X, np.argmax(y, axis=1))

    def predict(self, inputs: Sequence[np.ndarray]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            inputs (Sequence[np.ndarray]): ``(x_context, x_future)`` windows.

        Returns:
            np.ndarray: Class probabilities per sample.
        """
        x_context, x_future = inputs
        x_context_flat = x_context.reshape(x_context.shape[0], -1)
        x_future_flat = x_future.reshape(x_future.shape[0], -1)
        X = np.concatenate([x_context_flat, x_future_flat], axis=1)
        preds = self.model.predict_proba(X)
        return preds

    def save(self, filepath: str) -> None:
        """Pickle the wrapper to ``filepath``.

        Args:
            filepath (str): Destination path.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> "XGBClassifierWrapper":
        """Load a pickled wrapper.

        Args:
            filepath (str): Source path.

        Returns:
            XGBClassifierWrapper: The unpickled wrapper.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def build_xgboost_classifier(input_context_shape: Tuple[int, ...], input_future_shape: Tuple[int, ...], num_models: int, iteration_params: Dict[str, Any]) -> XGBClassifierWrapper:
    """Create an :class:`XGBClassifierWrapper`.

    Args:
        input_context_shape (Tuple[int, ...]): Per-sample encoder window shape.
        input_future_shape (Tuple[int, ...]): Per-sample decoder window shape.
        num_models (int): Number of candidate models.
        iteration_params (Dict[str, Any]): Iteration hyperparameters.

    Returns:
        XGBClassifierWrapper: The wrapper after its dummy fit.
    """
    return XGBClassifierWrapper(num_models, input_context_shape, input_future_shape, iteration_params)
