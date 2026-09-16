"""Loss functions for the Seq2Seq inflow forecaster.

All losses accept ``y_true`` with shape ``(batch, steps)`` or
``(batch, steps, 1)`` and ``y_pred`` with shape ``(batch, steps, 1)``, and
return a scalar ``tf.Tensor`` (mean over all elements).
"""
from __future__ import annotations

from typing import Optional

import tensorflow as tf


def quantile_pinball_loss(y_true: tf.Tensor, y_pred: tf.Tensor, tau: float = 0.9) -> tf.Tensor:
    """Quantile (pinball) loss.

    ``L_tau(y, y_hat) = max(tau * (y - y_hat), (tau - 1) * (y - y_hat))``.
    A quantile ``tau > 0.5`` penalizes under-prediction more than
    over-prediction.

    Args:
        y_true (tf.Tensor): Observed target, shape ``(batch, steps)`` or
            ``(batch, steps, 1)``.
        y_pred (tf.Tensor): Predictions, shape ``(batch, steps, 1)``.
        tau (float): Target quantile in ``(0, 1)``.

    Returns:
        tf.Tensor: Scalar mean pinball loss.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # y_true is the full target sequence (batch, steps). Expand on the LAST
    # axis to match y_pred (batch, steps, 1) element-wise; expanding axis=1
    # would broadcast to (batch, steps, steps) for steps > 1.
    if tf.rank(y_true) == 2:
        y_true = tf.expand_dims(y_true, axis=-1)

    e = y_true - y_pred  # the standard (y - y_hat) sign convention for pinball
    loss = tf.maximum(tau * e, (tau - 1.0) * e)  # shape (batch, steps, 1)
    return tf.reduce_mean(loss)


def pinball_from_penalty(y_true: tf.Tensor, y_pred: tf.Tensor, penalty_factor: float = 2.0) -> tf.Tensor:
    """Pinball loss with the quantile derived from an integer penalty.

    ``tau = clip((penalty + 1) / (penalty + 2), 0.5, 0.99)``, e.g.
    p=0 -> 0.500, p=1 -> 0.667, p=4 -> 0.833, p=9 -> 0.909.

    Args:
        y_true (tf.Tensor): Observed target.
        y_pred (tf.Tensor): Predictions.
        penalty_factor (float): Asymmetry penalty ``p >= 0``.

    Returns:
        tf.Tensor: Scalar mean pinball loss at the derived quantile.
    """
    # tf.function does not handle Python floats well inside the graph, so the
    # penalty is cast to a tensor before deriving tau.
    pf = tf.cast(penalty_factor, tf.float32)
    tau = tf.clip_by_value((pf + 1.0) / (pf + 2.0), 0.5, 0.99)
    # quantile_pinball_loss expects a Python float tau.
    tau_val = float(tau.numpy()) if hasattr(tau, "numpy") else float(tau)
    return quantile_pinball_loss(y_true, y_pred, tau=tau_val)


def custom_loss_mae(y_true: tf.Tensor, y_pred: tf.Tensor, penalty_factor: Optional[float] = 0.20) -> tf.Tensor:
    """Asymmetric MAE that penalizes under-prediction more heavily.

    Args:
        y_true (tf.Tensor): Observed target.
        y_pred (tf.Tensor): Predictions.
        penalty_factor (Optional[float]): Extra weight applied to errors where
            the prediction is below the observed value.

    Returns:
        tf.Tensor: Scalar mean penalized absolute error.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Idempotent with respect to rank: train_step already guarantees rank-3 y
    # (see models.training._ensure_loss_shape). Expanding unconditionally
    # would produce rank 4 and break broadcasting, so only rank-2 inputs (e.g.
    # direct calls from tests) are expanded.
    if tf.rank(y_true) == 2:  # (batch, steps) -> (batch, steps, 1)
        y_true = tf.expand_dims(y_true, axis=-1)
    y_expanded = y_true

    error = y_expanded - y_pred
    abs_error = tf.abs(error)

    # Positive error means the prediction is below the observed value.
    penalized_error = tf.where(error > 0, abs_error * (1 + penalty_factor), abs_error)
    return tf.reduce_mean(penalized_error)


def original_mae(y_true: tf.Tensor, y_pred: tf.Tensor, penalty_factor: Optional[float] = 0.20) -> tf.Tensor:
    """Symmetric mean absolute error.

    Args:
        y_true (tf.Tensor): Observed target.
        y_pred (tf.Tensor): Predictions.
        penalty_factor (Optional[float]): Unused; kept so that every loss in the
            registry shares a common call signature.

    Returns:
        tf.Tensor: Scalar mean absolute error.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Idempotent with respect to rank; see custom_loss_mae.
    if tf.rank(y_true) == 2:  # (batch, steps) -> (batch, steps, 1)
        y_true = tf.expand_dims(y_true, axis=-1)
    y_expanded = y_true

    error = y_expanded - y_pred
    abs_error = tf.abs(error)

    return tf.reduce_mean(abs_error)


def quantile_pinball_weighted(y_true: tf.Tensor, y_pred: tf.Tensor, tau: float = 0.9, k_high: float = 3.0) -> tf.Tensor:
    """Pinball loss with a larger weight on high flows.

    ``w = 1 + k_high * y_true``, assuming a normalized target in ``[0, 1]``.

    Args:
        y_true (tf.Tensor): Observed (normalized) target.
        y_pred (tf.Tensor): Predictions.
        tau (float): Target quantile in ``(0, 1)``.
        k_high (float): Slope of the high-flow weight.

    Returns:
        tf.Tensor: Scalar weighted mean pinball loss.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if tf.rank(y_true) == 2:  # (batch, steps) -> (batch, steps, 1); see quantile_pinball_loss
        y_true = tf.expand_dims(y_true, axis=-1)

    e = y_true - y_pred
    base = tf.maximum(tau * e, (tau - 1.0) * e)  # (batch, steps, 1)

    w = 1.0 + k_high * y_true  # (batch, steps, 1)
    return tf.reduce_mean(w * base)


def pinball_weighted_from_penalty(y_true: tf.Tensor, y_pred: tf.Tensor, penalty_factor: float = 2.0, k_high: float = 3.0) -> tf.Tensor:
    """High-flow weighted pinball loss with the quantile derived from a penalty.

    Uses the same ``tau = (penalty + 1) / (penalty + 2)`` mapping as
    :func:`pinball_from_penalty`.

    Args:
        y_true (tf.Tensor): Observed (normalized) target.
        y_pred (tf.Tensor): Predictions.
        penalty_factor (float): Asymmetry penalty ``p >= 0``.
        k_high (float): Slope of the high-flow weight.

    Returns:
        tf.Tensor: Scalar weighted mean pinball loss.
    """
    pf = tf.cast(penalty_factor, tf.float32)
    tau = tf.clip_by_value((pf + 1.0) / (pf + 2.0), 0.5, 0.99)
    tau_val = float(tau.numpy()) if hasattr(tau, "numpy") else float(tau)
    return quantile_pinball_weighted(y_true, y_pred, tau=tau_val, k_high=k_high)
