"""Sliding-window construction for the Seq2Seq encoder/decoder inputs."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ...log.log_config import get_logger

logger = get_logger()


def sliding_window(
    historicos: pd.DataFrame,
    predicciones: pd.DataFrame,
    labels: pd.Series,
    output_column: str,
    input_width: int,
    offset: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build encoder inputs, decoder inputs and multi-step targets.

    The target ``y`` of each window is the COMPLETE sequence
    ``[Qe(t+1), ..., Qe(t+offset+1)]`` (``offset + 1`` steps), so that the
    training step can compute a true sequence loss.

    The decoder carries, step by step, the real rainfall forecast of the day
    matching each target element: step 0 (t+1, 24 h) -> ``pred_l/m2``;
    step 1 (t+2, 48 h) -> ``pred_l/m2_2d``; step 2 (t+3, 72 h) ->
    ``pred_l/m2_3d``; later steps repeat ``pred_l/m2_3d`` (no forecast exists
    beyond 72 h). Each decoder row is ``[last Qe] + [rain forecast] * 4``.

    Args:
        historicos (pd.DataFrame): Historical encoder features.
        predicciones (pd.DataFrame): Rainfall forecast columns.
        labels (pd.Series): Target series (``Qe``).
        output_column (str): Name of the target column in ``historicos``.
        input_width (int): Encoder context length (days).
        offset (int): Lead-time offset; the decoder runs ``offset + 1`` steps.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Encoder windows
        ``(N, input_width, features)``, decoder inputs
        ``(N, offset + 1, 5)`` and targets ``(N, offset + 1)``.

    Raises:
        ValueError: If ``offset`` is negative.
    """
    if offset < 0:
        raise ValueError("offset cannot be negative")

    RAIN_COLS = ["pred_l/m2", "pred_l/m2_2d", "pred_l/m2_3d"]
    n_steps = offset + 1

    encoder_inputs, decoder_inputs, y = [], [], []
    n_rows = len(historicos)

    max_loops = n_rows - input_width - offset

    for i in range(max_loops):
        try:
            contexto_historico = historicos.iloc[i : i + input_width].values
            ultimo_qe = historicos.iloc[i + input_width - 1][output_column]
            pred_row = predicciones.iloc[i + input_width - 1]

            decoder_entrada = np.array([
                [ultimo_qe] + [pred_row[RAIN_COLS[min(k, len(RAIN_COLS) - 1)]]] * 4
                for k in range(n_steps)
            ])

            target_idx = i + input_width  # first target step (t+1)
            ventana_y = labels.iloc[target_idx : target_idx + n_steps].values

            encoder_inputs.append(contexto_historico)
            decoder_inputs.append(decoder_entrada)
            y.append(ventana_y)
        except Exception as e:
            logger.error(f"Error while processing row {i}: {e}")
            raise

    return np.array(encoder_inputs), np.array(decoder_inputs), np.array(y)


def prepare_model_inputs(
    train_norm: pd.DataFrame,
    val_norm: Optional[pd.DataFrame],
    test_norm: pd.DataFrame,
    variable_salida: str,
    historicos: List[str],
    predicciones: List[str],
    contextos: int,
    offset: int,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]],
    Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[List[pd.Timestamp]]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]
]:
    """Build encoder/decoder/target windows for train, validation and test.

    Windows never cross a date gap: each continuous block of daily records is
    windowed separately. The three partitions are processed in complete
    isolation, since the dataset generator already prepared each one.

    Args:
        train_norm (pd.DataFrame): Normalized training partition with ``Fecha``.
        val_norm (Optional[pd.DataFrame]): Normalized validation partition.
        test_norm (pd.DataFrame): Normalized test partition with ``Fecha``.
        variable_salida (str): Target column name.
        historicos (List[str]): Encoder feature columns.
        predicciones (List[str]): Rainfall forecast columns.
        contextos (int): Encoder context length (days).
        offset (int): Lead-time offset.

    Returns:
        Tuple: ``(train, val, test)``, each ``(x_enc, x_dec, y, dates)`` with one
        date per window (the date of the final evaluated horizon). ``val`` is
        ``(None, None, None, None)`` when ``val_norm`` is ``None`` or empty.
    """
    def _build_windows_for_df(df: Optional[pd.DataFrame]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[List[pd.Timestamp]]]:
        """Window one partition, block by block.

        Args:
            df (Optional[pd.DataFrame]): Partition to window.

        Returns:
            Tuple: ``(x_enc, x_dec, y, dates)``; all ``None`` for a missing or
            empty partition, and empty arrays when no block is long enough.
        """
        if df is None or df.empty:
            return None, None, None, None

        df = df.copy()

        # Ensure Fecha is a datetime before any date arithmetic.
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        df = df.sort_values(by="Fecha")

        # 1. Detect gaps longer than one day so that windows never cross them.
        df['gap'] = df['Fecha'].diff() > pd.Timedelta(days=1)
        df['chunk_id'] = df['gap'].cumsum()

        X_enc_list, X_dec_list, y_list, fechas_list = [], [], [], []

        # 2. Window each continuous block separately.
        for _, chunk_df in df.groupby('chunk_id'):
            if len(chunk_df) < contextos + offset:
                continue  # block too short for a single window

            fechas_chunk = chunk_df["Fecha"].values
            chunk_data = chunk_df.drop(columns=["Fecha", "gap", "chunk_id"], errors="ignore")

            x_enc, x_dec, y = sliding_window(
                historicos=chunk_data[historicos],
                predicciones=chunk_data[predicciones],
                labels=chunk_data[variable_salida],
                output_column=variable_salida,
                input_width=contextos,
                offset=offset
            )

            # One date per WINDOW (not per step): the date of the final
            # evaluated horizon, t+offset+1. y[k] holds the full sequence
            # [t+1..t+offset+1], but the rest of the pipeline (dates in the
            # prediction .npz files, the classifier's anti-shuffle check, ...)
            # expects one date per window, that of the evaluated target.
            target_start_idx = contextos + offset
            y_fechas = fechas_chunk[target_start_idx : target_start_idx + len(y)]

            # Temporal-alignment sanity check, logged for the first window of
            # the first usable block only.
            if len(y) > 0 and len(X_enc_list) == 0:
                fecha_inicio_X = fechas_chunk[0]
                fecha_fin_X = fechas_chunk[contextos - 1]
                fecha_target_y = y_fechas[0]

                # Actual distance in days between the last input day and the target.
                distancia_dias = (pd.to_datetime(fecha_target_y) - pd.to_datetime(fecha_fin_X)).days

                logger.info("\n" + "="*50)
                logger.info("WINDOW AUDIT (first record of the block):")
                logger.info(f"  [INPUT X] History start: {pd.to_datetime(fecha_inicio_X).date()}")
                logger.info(f"  [INPUT X] History end:   {pd.to_datetime(fecha_fin_X).date()} (issue date)")
                logger.info(f"  [TARGET y] Forecast date: {pd.to_datetime(fecha_target_y).date()}")
                logger.info(f"  -> Forecast distance:     {distancia_dias} days ahead (48h={distancia_dias==2})")

                # Raw values for confirmation: y[0] is the sequence
                # [t+1..t+offset+1] and its last element matches fecha_target_y.
                val_fin_X = chunk_data.iloc[contextos - 1][variable_salida]
                val_target_y = y[0][-1]
                logger.info(f"  [VALUES] Qe on {pd.to_datetime(fecha_fin_X).date()}: {val_fin_X:.4f}")
                logger.info(f"  [VALUES] Qe on {pd.to_datetime(fecha_target_y).date()}: {val_target_y:.4f}")
                logger.info("="*50 + "\n")

            if len(y) > 0:
                X_enc_list.append(x_enc)
                X_dec_list.append(x_dec)
                y_list.append(y)
                fechas_list.extend(y_fechas)

        if not X_enc_list:
            return np.array([]), np.array([]), np.array([]), []

        return np.vstack(X_enc_list), np.vstack(X_dec_list), np.vstack(y_list), list(fechas_list)

    # The three partitions are windowed in complete isolation.
    train_res = _build_windows_for_df(train_norm)
    val_res = _build_windows_for_df(val_norm)
    test_res = _build_windows_for_df(test_norm)

    return train_res, val_res, test_res
