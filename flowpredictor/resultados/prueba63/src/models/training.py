"""Training loop of the Seq2Seq LSTM forecaster: loss dispatch, decoder
unrolling, gradient step and early stopping."""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from ...log.log_config import get_logger
from . import losses as _losses
from .losses import quantile_pinball_loss, quantile_pinball_weighted

logger = get_logger()


PARAM_NAMES = {"penalty_factor", "penalty", "alpha", "lambda_penalty"}

PARAM_MAP = {
    "penalty_factor": "penalty",
    "penalty": "penalty",
    "alpha": "penalty",
    "lambda_penalty": "penalty",
    "k_high": "k_high",
    "delta": "delta",  # reserved for an asymmetric pseudo-Huber loss
    "tau": "tau",  # for losses that accept tau directly
}


LOSS_REGISTRY = {
    "original_mae": getattr(_losses, "original_mae", None),
    "custom_loss_mae": getattr(_losses, "custom_loss_mae", None),
    "pinball_from_penalty": getattr(_losses, "pinball_from_penalty", None),
    "pinball_weighted_from_penalty": getattr(_losses, "pinball_weighted_from_penalty", None),
    "quantile_pinball_loss": getattr(_losses, "quantile_pinball_loss", None),
    "quantile_pinball_weighted": getattr(_losses, "quantile_pinball_weighted", None),
}


def _get_loss_fn(name: str) -> Callable[..., tf.Tensor]:
    """Look up a loss function by name.

    Args:
        name (str): Key of :data:`LOSS_REGISTRY`.

    Returns:
        Callable[..., tf.Tensor]: The registered loss function.

    Raises:
        ValueError: If the name is not registered.
    """
    fn = LOSS_REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"[LOSS] Unknown: '{name}'. Add it to LOSS_REGISTRY in training.py or to config.py")
    return fn


def _to_float_or_zero(x: Any) -> float:
    """Convert a scalar, or the first element of a list/tuple, to float.

    Args:
        x (Any): Value to convert.

    Returns:
        float: The converted value, or ``0.0`` if conversion fails.
    """
    try:
        if isinstance(x, (list, tuple)):
            x = x[0] if len(x) > 0 else 0.0
        return float(x)
    except Exception:
        return 0.0


def compute_loss_by_name(loss_name: str, y_true: tf.Tensor, y_pred: tf.Tensor, penalty_value: Optional[float] = None, k_high: Optional[float] = None, tau: Optional[float] = None) -> tf.Tensor:
    """Evaluate a registered loss, passing the arguments it expects.

    For the pinball losses, an explicit ``tau`` (injected by
    :func:`train_model`) takes precedence over deriving it from the penalty.

    Args:
        loss_name (str): Key of :data:`LOSS_REGISTRY`.
        y_true (tf.Tensor): Observed target.
        y_pred (tf.Tensor): Predictions.
        penalty_value (Optional[float]): Asymmetry penalty.
        k_high (Optional[float]): High-flow weight slope.
        tau (Optional[float]): Explicit quantile.

    Returns:
        tf.Tensor: Scalar loss value.

    Raises:
        ValueError: If ``loss_name`` is not registered.
    """
    fn = _get_loss_fn(loss_name)
    logger.debug(f"[LOSS] Using '{loss_name}' with penalty={penalty_value}, k_high={k_high}, tau={tau}")
    if loss_name == "original_mae":
        return fn(y_true, y_pred)

    if loss_name == "custom_loss_mae":
        return fn(y_true, y_pred, penalty_factor=float(penalty_value or 0.0))

    if loss_name == "pinball_from_penalty":
        # If train_model already injected the exact tau into iteration_params,
        # use it directly (quantile_pinball_loss) instead of letting
        # pinball_from_penalty re-derive it from the penalty: a single source of
        # truth for tau, never the 0.9 default.
        if tau is not None:
            return quantile_pinball_loss(y_true, y_pred, tau=float(tau))
        return fn(y_true, y_pred, penalty_factor=float(penalty_value or 0.0))

    if loss_name == "pinball_weighted_from_penalty":
        if tau is not None:
            return quantile_pinball_weighted(y_true, y_pred, tau=float(tau), k_high=float(k_high or 0.0))
        return fn(y_true, y_pred, penalty_factor=float(penalty_value or 0.0),
                  k_high=float(k_high or 0.0))

    if loss_name == "quantile_pinball_loss":
        return fn(y_true, y_pred, tau=float(tau or 0.9))

    if loss_name == "quantile_pinball_weighted":
        return fn(y_true, y_pred, tau=float(tau or 0.9), k_high=float(k_high or 0.0))

    return fn(y_true, y_pred)


def compute_loss(loss_fn: Callable[..., tf.Tensor], y_true: tf.Tensor, y_pred: tf.Tensor, penalty_value: Optional[float] = None, iteration_params: Optional[Dict[str, Any]] = None) -> tf.Tensor:
    """Call a loss, injecting penalty/k_high/tau/delta when its signature accepts them.

    Args:
        loss_fn (Callable[..., tf.Tensor]): Loss function, ``functools.partial``
            or Keras ``Loss`` instance.
        y_true (tf.Tensor): Observed target.
        y_pred (tf.Tensor): Predictions.
        penalty_value (Optional[float]): Asymmetry penalty, passed under the
            first accepted name among ``penalty_factor``, ``penalty``,
            ``alpha`` and ``lambda_penalty``.
        iteration_params (Optional[Dict[str, Any]]): Source of ``k_high``,
            ``tau`` and ``delta``.

    Returns:
        tf.Tensor: Scalar loss value.
    """
    base_fn = loss_fn
    if hasattr(loss_fn, "__wrapped__"):
        base_fn = loss_fn.__wrapped__
    if isinstance(loss_fn, functools.partial):
        base_fn = loss_fn.func

    # Keras Loss instances take (y_true, y_pred) only.
    if isinstance(loss_fn, tf.keras.losses.Loss):
        return loss_fn(y_true, y_pred)

    if not callable(loss_fn):
        return loss_fn(y_true, y_pred)

    try:
        params = set(inspect.signature(base_fn).parameters.keys())
    except Exception:
        return loss_fn(y_true, y_pred)

    kw = {}
    if penalty_value is not None:
        for pname in ("penalty_factor", "penalty", "alpha", "lambda_penalty"):
            if pname in params:
                kw[pname] = penalty_value
                break

    if iteration_params:
        if "k_high" in params and ("k_high" in iteration_params):
            kw["k_high"] = iteration_params["k_high"]
        if "tau" in params and ("tau" in iteration_params):
            kw["tau"] = iteration_params["tau"]
        if "delta" in params and ("delta" in iteration_params):
            kw["delta"] = iteration_params["delta"]

    return loss_fn(y_true, y_pred, **kw)


def _unroll_decoder(encoder_model: Model, decoder_model: Model, x_enc: tf.Tensor, x_dec: tf.Tensor, n_steps: int, training: bool, projector_model: Optional[Model] = None) -> tf.Tensor:
    """Unroll the decoder ``n_steps`` times and return every prediction.

    ``n_steps = offsets + 1``, one per element of the target sequence
    ``[Qe(t+1), ..., Qe(t+offsets+1)]``. Each step uses the precomputed real
    rainfall forecast in ``x_dec`` (see ``preprocessing.sliding_window``); the
    streamflow channel of step ``k >= 1`` is the model's own prediction from
    step ``k - 1``.

    ``projector_model`` maps the encoder state (``encoder_units``) to the
    decoder width (``decoder_units``) ONCE, before the first step. From then on
    the decoder feeds back its own state without any projection, so that no
    linear transformation enters the recurrence and the LSTM Constant Error
    Carousel is preserved.

    Args:
        encoder_model (Model): Encoder.
        decoder_model (Model): Single-step decoder.
        x_enc (tf.Tensor): Encoder windows ``(batch, context, features)``.
        x_dec (tf.Tensor): Decoder inputs ``(batch, n_steps, features)``.
        n_steps (int): Number of decoder steps.
        training (bool): Keras ``training`` flag.
        projector_model (Optional[Model]): State projector for asymmetric widths.

    Returns:
        tf.Tensor: Predictions of shape ``(batch, n_steps, 1)``.
    """
    state_h, state_c = encoder_model(x_enc, training=training)
    if projector_model is not None:
        state_h, state_c = projector_model([state_h, state_c], training=training)
    current_input = x_dec[:, :1, :]
    predictions = []
    for step in range(n_steps):
        pred, state_h, state_c = decoder_model([current_input, state_h, state_c], training=training)
        predictions.append(pred)
        if step < x_dec.shape[1] - 1:
            current_input = x_dec[:, step + 1:step + 2, :]
        current_input = tf.concat(
            [tf.cast(pred, tf.float32), tf.cast(current_input[:, :, 1:], tf.float32)],
            axis=-1
        )
    return tf.concat(predictions, axis=1)


def _one_batch_preds(encoder_model: Model, decoder_model: Model, dataset: tf.data.Dataset, iteration_params: Dict[str, Any], projector_model: Optional[Model] = None) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:
    """Take one batch and return ``(y, predictions)`` with the training unroll.

    Args:
        encoder_model (Model): Encoder.
        decoder_model (Model): Single-step decoder.
        dataset (tf.data.Dataset): Batched ``(x_enc, x_dec, y)`` dataset.
        iteration_params (Dict[str, Any]): Iteration hyperparameters (``offsets``).
        projector_model (Optional[Model]): State projector for asymmetric widths.

    Returns:
        Tuple[Optional[tf.Tensor], Optional[tf.Tensor]]: ``(y, predictions)``,
        or ``(None, None)`` for an empty dataset.
    """
    n_steps = int(iteration_params["offsets"]) + 1
    for x_enc_b, x_dec_b, y_b in dataset.take(1):
        predictions_b = _unroll_decoder(encoder_model, decoder_model, x_enc_b, x_dec_b, n_steps, training=False, projector_model=projector_model)
        y_b = _ensure_loss_shape(y_b, predictions_b)
        return y_b, predictions_b
    return None, None


def _ensure_loss_shape(y: tf.Tensor, predictions: tf.Tensor) -> tf.Tensor:
    """Guarantee that ``y`` and ``predictions`` share shape ``(batch, steps, 1)``.

    A mismatch here (e.g. a rank-2 ``y``, or a different number of steps)
    would silently broadcast to ``(batch, steps, steps)`` instead of comparing
    element-wise; this guard makes that bug impossible rather than unlikely.

    Args:
        y (tf.Tensor): Target, shape ``(batch, steps)`` or ``(batch, steps, 1)``.
        predictions (tf.Tensor): Predictions, shape ``(batch, steps, 1)``.

    Returns:
        tf.Tensor: ``y`` with shape ``(batch, steps, 1)``.

    Raises:
        tf.errors.InvalidArgumentError: If the shapes are incompatible.
    """
    if tf.rank(y) == 2:  # (batch, steps)
        y = tf.expand_dims(y, axis=-1)  # -> (batch, steps, 1)
    tf.debugging.assert_shapes(
        [(y, ["B", "S", "1"]), (predictions, ["B", "S", "1"])],
        message="Corrupted dimensions before the loss",
    )
    return y


def train_step(
    encoder_model: Model, decoder_model: Model,
    x_enc: tf.Tensor, x_dec: tf.Tensor, y: tf.Tensor,
    optimizer: Optimizer, iteration_params: Dict[str, Any],
    projector_model: Optional[Model] = None,
) -> tf.Tensor:
    """Run one gradient step with a full sequence loss.

    ``y`` is the complete target sequence ``[Qe(t+1), ..., Qe(t+offsets+1)]``,
    so the decoder is unrolled exactly ``offsets + 1`` times and every
    generated step is supervised by its own target. The pinball quantile is
    static across epochs (no curriculum): a per-epoch moving quantile would
    break the calibration guarantee of quantile regression.

    Args:
        encoder_model (Model): Encoder.
        decoder_model (Model): Single-step decoder.
        x_enc (tf.Tensor): Encoder windows.
        x_dec (tf.Tensor): Decoder inputs.
        y (tf.Tensor): Target sequence.
        optimizer (Optimizer): Keras optimizer.
        iteration_params (Dict[str, Any]): Uses ``offsets``, ``loss_name``,
            ``penalty``, ``k_high`` and ``tau``.
        projector_model (Optional[Model]): State projector for asymmetric widths.

    Returns:
        tf.Tensor: Scalar loss of the step.

    Raises:
        AssertionError: If ``iteration_params["loss_name"]`` is not a string.
    """
    n_steps = int(iteration_params["offsets"]) + 1
    with tf.GradientTape() as tape:
        predictions = _unroll_decoder(encoder_model, decoder_model, x_enc, x_dec, n_steps, training=True, projector_model=projector_model)
        y = _ensure_loss_shape(y, predictions)

        effective_penalty = float(iteration_params.get("penalty", 0.0))
        assert isinstance(iteration_params["loss_name"], str), "loss_name must be a string"
        logger.debug(f"[TRAIN STEP] k_high={iteration_params.get('k_high')} | effective_penalty={effective_penalty} | tau={iteration_params.get('tau')}")

        loss_value = compute_loss_by_name(
            iteration_params["loss_name"],
            y,
            predictions,
            penalty_value=effective_penalty,
            k_high=iteration_params.get("k_high"),
            tau=iteration_params.get("tau"),
        )

    variables = encoder_model.trainable_variables + decoder_model.trainable_variables
    if projector_model is not None:
        variables = variables + projector_model.trainable_variables
    gradients = tape.gradient(loss_value, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss_value


def train_model(
    encoder_model: Model,
    decoder_model: Model,
    x_train_encoder: np.ndarray,
    x_train_decoder: np.ndarray,
    y_train: np.ndarray,
    x_val_encoder: Optional[np.ndarray],
    x_val_decoder: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    iteration_params: Dict[str, Any],
    projector_model: Optional[Model] = None,
) -> Dict[str, List[float]]:
    """Train the Seq2Seq model with Adam and optional early stopping.

    Early stopping is opt-in (``early_stopping_patience``). It monitors the
    validation loss when validation data is given and the training loss
    otherwise; the current pipeline passes no validation split.

    Args:
        encoder_model (Model): Encoder.
        decoder_model (Model): Single-step decoder.
        x_train_encoder (np.ndarray): Training encoder windows.
        x_train_decoder (np.ndarray): Training decoder inputs.
        y_train (np.ndarray): Training target sequences.
        x_val_encoder (Optional[np.ndarray]): Validation encoder windows.
        x_val_decoder (Optional[np.ndarray]): Validation decoder inputs.
        y_val (Optional[np.ndarray]): Validation target sequences.
        iteration_params (Dict[str, Any]): Uses ``lr``, ``batch_size``,
            ``max_epochs``, ``offsets``, ``loss_name``, ``penalty``, ``k_high``,
            ``early_stopping_patience`` and ``early_stopping_min_delta``; for
            pinball losses ``tau`` is written into it.
        projector_model (Optional[Model]): State projector for asymmetric widths.

    Returns:
        Dict[str, List[float]]: Per-epoch ``train_loss`` and ``val_loss`` history.
    """
    optimizer = tf.optimizers.Adam(learning_rate=iteration_params['lr'])
    history = {"train_loss": [], "val_loss": []}

    early_stopping_patience = iteration_params.get("early_stopping_patience")
    early_stopping_min_delta = float(iteration_params.get("early_stopping_min_delta", 1e-4))
    best_monitored_loss = float("inf")
    epochs_without_improvement = 0

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train_encoder, tf.float32),
         tf.cast(x_train_decoder, tf.float32),
         tf.cast(y_train, tf.float32))
    ).batch(iteration_params["batch_size"])
    if x_val_encoder is not None:
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(x_val_encoder, tf.float32),
             tf.cast(x_val_decoder, tf.float32),
             tf.cast(y_val, tf.float32))
        ).batch(iteration_params["batch_size"])
    else:
        val_dataset = None

    for epoch in range(iteration_params["max_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{iteration_params['max_epochs']}/{iteration_params['loss_name']}/k={iteration_params.get('k_high')}/penalty={iteration_params.get('penalty')}")
        train_loss = 0.0
        val_loss = 0.0

        # No curriculum: the penalty (and therefore tau for pinball losses) is
        # the STATIC hyperparameter of iteration_params, identical in the first
        # and last epoch. A moving per-epoch tau is not a valid quantile.
        static_penalty = float(iteration_params.get("penalty", 0.0))
        if iteration_params.get("loss_name") in ("pinball_from_penalty", "pinball_weighted_from_penalty"):
            static_tau = min(0.99, (static_penalty + 1.0) / (static_penalty + 2.0))
            # iteration_params['tau'] becomes the single source of truth used by
            # train_step/compute_loss_by_name for the pinball losses.
            iteration_params["tau"] = static_tau
            logger.debug(f"[PINBALL] epoch={epoch} penalty={static_penalty:.3f} tau={static_tau:.3f} k_high={iteration_params.get('k_high')}")
        if epoch == 0:
            y_smoke, preds_smoke = _one_batch_preds(encoder_model, decoder_model, train_dataset, iteration_params, projector_model=projector_model)
            if (y_smoke is not None) and (preds_smoke is not None):
                if iteration_params.get("loss_name") in ("pinball_from_penalty", "pinball_weighted_from_penalty"):
                    l_base = quantile_pinball_loss(y_smoke, preds_smoke, tau=0.9).numpy()
                    l_w = quantile_pinball_weighted(y_smoke, preds_smoke, tau=0.9, k_high=3.0).numpy()
                    logger.info(f"[PINBALL] base={l_base:.6f} | weighted={l_w:.6f}")
                else:
                    logger.info(f"[MEAN ABS ERROR] loss={iteration_params.get('loss_name')}")
                yhat = preds_smoke.numpy()
                logger.info(f"[YHAT] mean={yhat.mean():.4f} std={yhat.std():.4f} max={yhat.max():.4f}")

        for x_enc, x_dec, y in train_dataset:
            loss_value = train_step(
                encoder_model,
                decoder_model,
                x_enc,
                x_dec,
                y,
                optimizer,
                iteration_params,
                projector_model=projector_model,
            )
            train_loss += float(loss_value.numpy())

        if val_dataset is not None:
            n_steps = int(iteration_params["offsets"]) + 1
            for x_enc, x_dec, y in val_dataset:
                # Same unroll and sequence loss as train_step.
                predictions = _unroll_decoder(encoder_model, decoder_model, x_enc, x_dec, n_steps, training=False, projector_model=projector_model)
                y = _ensure_loss_shape(y, predictions)

                val_loss += compute_loss_by_name(
                    iteration_params["loss_name"],
                    y, predictions,
                    penalty_value=iteration_params.get("penalty", 0.0),
                    k_high=iteration_params.get("k_high", None),
                    tau=iteration_params.get("tau", None),
                ).numpy()

        train_loss /= len(train_dataset)
        history["train_loss"].append(train_loss)
        if val_dataset is not None:
            val_loss /= len(val_dataset)
            history["val_loss"].append(val_loss)
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            val_loss = "N/A"
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: N/A")

        if early_stopping_patience is not None:
            monitored = val_loss if isinstance(val_loss, float) else train_loss
            if monitored < best_monitored_loss - early_stopping_min_delta:
                best_monitored_loss = monitored
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= int(early_stopping_patience):
                logger.info(
                    f"[EARLY STOPPING] No improvement in {early_stopping_patience} epochs "
                    f"(best loss={best_monitored_loss:.6f}); stopping at epoch {epoch + 1}/{iteration_params['max_epochs']}."
                )
                break

    return history
