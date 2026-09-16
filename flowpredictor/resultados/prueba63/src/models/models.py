"""Keras building blocks of the Seq2Seq LSTM forecaster (encoder, decoder and
the optional encoder-to-decoder state projector)."""
from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Dense, Input, LSTM


def build_encoder(input_shape: Tuple[int, int], iteration_params: Dict[str, Any]) -> Tuple[Model, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Build the LSTM encoder.

    Seeds TensorFlow, NumPy and Python's ``random`` from
    ``iteration_params["seed"]`` before creating any layer.

    Args:
        input_shape (Tuple[int, int]): Encoder input shape ``(steps, features)``.
        iteration_params (Dict[str, Any]): Iteration hyperparameters. Uses
            ``seed``, ``activation``, ``encoder_units`` and ``dropout``.

    Returns:
        Tuple[Model, tf.Tensor, tf.Tensor, tf.Tensor]: The encoder model, its
        input tensor, and the final hidden and cell state tensors.
    """
    seed = iteration_params.get("seed", 42)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    default_activation = 'tanh'
    activation = iteration_params.get("activation", default_activation)
    encoder_units = int(iteration_params.get("encoder_units", 64))

    encoder_inputs = Input(shape=input_shape, name="encoder_inputs")
    encoder_lstm = LSTM(
        encoder_units, return_state=True, activation=activation, dropout=0.2 if iteration_params["dropout"] else 0.0, name="encoder_lstm"
    )
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_model = Model(inputs=encoder_inputs, outputs=[state_h, state_c], name="encoder_model")
    return encoder_model, encoder_inputs, state_h, state_c


def build_decoder(
    input_shape: int,
    iteration_params: Dict[str, Any],
) -> Model:
    """Build the single-step LSTM decoder with explicit initial-state inputs.

    Seeds TensorFlow, NumPy and Python's ``random`` from
    ``iteration_params["seed"]`` before creating any layer.

    Args:
        input_shape (int): Number of decoder input features per step.
        iteration_params (Dict[str, Any]): Iteration hyperparameters. Uses
            ``seed``, ``activation``, ``decoder_units``, ``dropout`` and
            ``l2_options``.

    Returns:
        Model: Decoder model with inputs ``[decoder_inputs, state_h, state_c]``
        and outputs ``[prediction, state_h, state_c]``.
    """
    seed = iteration_params.get("seed", 42)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    default_activation = 'tanh'
    activation = iteration_params.get("activation", default_activation)
    decoder_units = int(iteration_params.get("decoder_units", 64))

    # The state inputs are declared with decoder_units: the decoder is called
    # repeatedly (_unroll_decoder / _predict_iterative) feeding back its OWN
    # state unchanged from one step to the next. Any Dense layer on state_c
    # inside that loop would add a matrix multiplication to the cell-state
    # recurrence and break the LSTM Constant Error Carousel. The bridge for
    # asymmetric encoder/decoder widths lives outside this model, in
    # build_state_projector(), and is applied once at the hand-over.
    decoder_inputs = Input(shape=(1, input_shape), name="decoder_inputs")
    decoder_state_input_h = Input(shape=(decoder_units,), name="decoder_state_h")
    decoder_state_input_c = Input(shape=(decoder_units,), name="decoder_state_c")

    decoder_lstm = LSTM(
        decoder_units,
        return_sequences=True,
        return_state=True,
        activation=activation,
        dropout=0.2 if iteration_params["dropout"] else 0.0,
        name="decoder_lstm"
    )
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=[decoder_state_input_h, decoder_state_input_c]
    )

    # Dense acts on the last axis independently of the leading axes, so on a
    # 1-step sequence it is equivalent to TimeDistributed(Dense). The output is
    # linear on purpose: a hard ReLU-style clip on the regression head can
    # permanently kill the gradient of the whole network ("dying ReLU"). The
    # normalized target lives in [0, 1] and the loss already drives predictions
    # towards it without forcing non-negativity in the layer itself.
    decoder_dense = Dense(
        1,
        kernel_regularizer=regularizers.L2(0.01) if iteration_params["l2_options"] else None,
        name="decoder_dense"
    )
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        inputs=[decoder_inputs, decoder_state_input_h, decoder_state_input_c],
        outputs=[decoder_outputs, state_h, state_c],
        name="decoder_model"
    )

    return decoder_model


def build_state_projector(encoder_units: int, decoder_units: int) -> Optional[Model]:
    """Build the encoder-to-decoder state bridge for asymmetric widths.

    Applied ONCE, at the encoder-to-decoder hand-over before the first
    autoregressive step and never inside the decoder loop, so that no linear
    transformation is introduced into the cell-state recurrence (see
    :func:`build_decoder`).

    Args:
        encoder_units (int): Encoder LSTM width.
        decoder_units (int): Decoder LSTM width.

    Returns:
        Optional[Model]: A model mapping ``[state_h, state_c]`` from
        ``encoder_units`` to ``decoder_units``, or ``None`` when both widths
        are equal and no bridge is needed.
    """
    if encoder_units == decoder_units:
        return None
    state_h_in = Input(shape=(encoder_units,), name="projector_state_h_in")
    state_c_in = Input(shape=(encoder_units,), name="projector_state_c_in")
    state_h_out = Dense(decoder_units, name="projector_state_h_out")(state_h_in)
    state_c_out = Dense(decoder_units, name="projector_state_c_out")(state_c_in)
    return Model(
        inputs=[state_h_in, state_c_in],
        outputs=[state_h_out, state_c_out],
        name="state_projector"
    )
