"""Single experimental iteration (one configuration, one test year).

An :class:`Iteration` owns every artifact of a run under a content-addressed
directory ``tests<N>/<hash_id>/``: the data bundle, the sliding windows, the
pretrained and fine-tuned Seq2Seq models, their predictions, the out-of-fold
predictions used to train the M5 meta-learner, and the meta-learner itself.
Every stage is resumable: it reuses artifacts already present on disk.
"""
from __future__ import annotations

import copy
import functools
import inspect
import json
import os
import pickle
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import hydroeval as he
except Exception:
    he = None

from ...log.log_config import get_logger
from ..config import CONFIG
from ..evaluation.metrics import compute_extreme_metrics, peak_timing_error
from ..models.losses import (
    custom_loss_mae,
    original_mae,
    pinball_from_penalty,
    pinball_weighted_from_penalty,
)
from ..models.models import build_decoder, build_encoder, build_state_projector
from ..models.randomforest import build_random_forest_classifier, build_random_forest_regressor
from ..models.training import train_model
from ..models.xgboost import build_xgboost_classifier
from .dataio import ensure_iteration_dataset, load_iteration_bundle, resolve_bundle_base
from .metrics import METRICS_REGISTRY
from .paths import IterationPaths, ensure_dirs
from .preprocessing import prepare_model_inputs
from .utils import _scalar, sha1_of_dict

# Seed of the published final run. A literal on purpose, NOT read from CONFIG:
# if it came from the TOML, changing the seed there would rewrite the hash of
# every already-validated set of weights instead of creating a new experiment.
# See its use in hash_payload (per-seed cache isolation).
_PUBLISHED_RUN_SEED = 92

# Loss name -> callable.
_LOSS_REGISTRY = {
    "original_mae": original_mae,
    "custom_loss_mae": custom_loss_mae,
    "pinball_from_penalty": pinball_from_penalty,
    "pinball_weighted_from_penalty": pinball_weighted_from_penalty,
}


def _make_loss_tag_simple(loss_name: str, k_high: Optional[float]) -> str:
    """Build the file-name tag of a loss, adding ``_k<k_high>`` for weighted losses.

    Args:
        loss_name (str): Loss name.
        k_high (Optional[float]): High-flow weight slope, if any.

    Returns:
        str: Tag such as ``pinball_from_penalty`` or
        ``pinball_weighted_from_penalty_k3``.
    """
    tag = loss_name
    if ("weighted" in loss_name) and (k_high not in (None, 0, 0.0)):
        v = float(k_high)
        tag += "_k" + (str(int(v)) if abs(v - int(v)) < 1e-9 else str(v).replace(".", "p"))
    return tag


def _resolve_loss_fn(loss_name: str) -> Callable[..., Any]:
    """Look up a loss callable by name.

    Args:
        loss_name (str): Key of ``_LOSS_REGISTRY``.

    Returns:
        Callable[..., Any]: The loss function.

    Raises:
        ValueError: If the name is not registered.
    """
    fn = _LOSS_REGISTRY.get(loss_name)
    if fn is None:
        raise ValueError(f"[LOSS] Unknown loss name: {loss_name}. "
                         f"Available: {list(_LOSS_REGISTRY.keys())}")
    return fn


def _asymmetric_loss_elementwise(y_true: np.ndarray, y_pred: np.ndarray, loss_name: str, penalty: float) -> np.ndarray:
    """Element-wise (unreduced) value of the configured asymmetric loss.

    Used by :meth:`Iteration.ensure_ensemble_labels` to compare candidate
    models time step by time step. Same formula as ``original_mae`` and
    ``pinball_from_penalty`` (``tau = (penalty + 1) / (penalty + 2)``,
    ``e = y_true - y_pred``, ``pinball = max(tau * e, (tau - 1) * e)``); the
    mean over a single element is a no-op, so each value equals the training
    loss evaluated on a batch of size one at that time step.

    Args:
        y_true (np.ndarray): Observed values, broadcastable against ``y_pred``.
        y_pred (np.ndarray): Predictions, e.g. ``(num_models, N)``.
        loss_name (str): ``original_mae`` or ``pinball_from_penalty``.
        penalty (float): Asymmetry penalty.

    Returns:
        np.ndarray: Element-wise loss, same shape as the broadcast inputs.

    Raises:
        NotImplementedError: For any other loss name.
    """
    if loss_name == "original_mae":
        return np.abs(y_true - y_pred)
    if loss_name == "pinball_from_penalty":
        tau = float(np.clip((penalty + 1.0) / (penalty + 2.0), 0.5, 0.99))
        e = y_true - y_pred
        return np.maximum(tau * e, (tau - 1.0) * e)
    raise NotImplementedError(
        f"[ENS] Asymmetric loss '{loss_name}' is not supported in ensure_ensemble_labels() "
        f"(supported: original_mae, pinball_from_penalty)."
    )


def _one_hot(idx: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode integer class indices.

    Args:
        idx (np.ndarray): Class indices, shape ``(N,)``.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: ``float32`` array of shape ``(N, num_classes)``.
    """
    out = np.zeros((idx.shape[0], num_classes), dtype=np.float32)
    out[np.arange(idx.shape[0]), idx.astype(int)] = 1.0
    return out


def _canonical_base(name: str) -> str:
    """Strip a trailing ``_train`` or ``_test`` from a prediction base name.

    Args:
        name (str): Base name, e.g. ``pinball_from_penalty_2_train``.

    Returns:
        str: The name without the subset suffix.
    """
    if name.endswith("_train"):
        return name[:-6]
    if name.endswith("_test"):
        return name[:-5]
    return name


def _basename_callable(fn: Any) -> str:
    """Sanitized base name of a loss function or loss object.

    Args:
        fn (Any): Function, ``functools.partial`` or object with a ``name``.

    Returns:
        str: Lower-case name with non-alphanumeric runs replaced by ``_``.
    """
    if hasattr(fn, "__name__"):
        name = fn.__name__
    elif hasattr(fn, "name"):
        name = fn.name
    elif isinstance(fn, functools.partial):
        name = _basename_callable(fn.func)
    else:
        name = type(fn).__name__
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _fmt_num(x: Any) -> str:
    """Format a number as a file-name-safe label.

    Examples: ``3 -> '3'``, ``2.5 -> '2p5'``, ``-0.75 -> 'm0p75'``.

    Args:
        x (Any): Number-like value.

    Returns:
        str: Safe label, or ``str(x)`` if ``x`` is not numeric.
    """
    try:
        xf = float(x)
    except Exception:
        return str(x)
    xi = int(round(xf))
    if abs(xf - xi) < 1e-9:
        return str(xi)
    s = f"{xf:.4g}".replace(".", "p").replace("-", "m")
    return s


def _loss_accepts(loss_fn: Callable[..., Any], arg: str) -> bool:
    """Whether a loss accepts a given keyword argument (e.g. ``k_high``, ``tau``).

    Args:
        loss_fn (Callable[..., Any]): Loss function or ``functools.partial``.
        arg (str): Argument name.

    Returns:
        bool: ``True`` if ``arg`` is in the loss signature.
    """
    base = loss_fn.func if isinstance(loss_fn, functools.partial) else loss_fn
    try:
        return arg in inspect.signature(base).parameters
    except Exception:
        return False


def _make_loss_tag(loss_fn: Callable[..., Any], loss_name_hint: Optional[str], it_p: Dict[str, Any]) -> str:
    """Build the file-name tag of a loss with its applicable extras.

    Base is ``loss_name_hint`` or the callable name; ``k<k_high>`` and
    ``t<tau>`` are appended when the loss accepts them and ``it_p`` sets them.

    Args:
        loss_fn (Callable[..., Any]): Loss function.
        loss_name_hint (Optional[str]): Preferred base name.
        it_p (Dict[str, Any]): Iteration hyperparameters.

    Returns:
        str: The tag.
    """
    base = (loss_name_hint or _basename_callable(loss_fn)).lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    extras = []
    if _loss_accepts(loss_fn, "k_high") and ("k_high" in it_p) and it_p["k_high"] not in (None, 0, 0.0):
        extras.append("k" + _fmt_num(it_p["k_high"]))
    if _loss_accepts(loss_fn, "tau") and ("tau" in it_p) and it_p["tau"] is not None:
        extras.append("t" + _fmt_num(it_p["tau"]))
    return base if not extras else f"{base}_{'_'.join(extras)}"


@dataclass
class IterationState:
    """Persisted state of an iteration.

    Attributes:
        params: Normalized iteration hyperparameters.
        ready_data: Whether the data bundle has been loaded.
    """
    params: Dict
    ready_data: bool = False


class Iteration:
    """One experimental iteration: a configuration evaluated on one test year.

    The hyperparameters are normalized (scalars extracted from single-element
    lists, defaults filled in) and hashed into ``hash_id``, which names the
    cache directory of every artifact of the iteration.

    Args:
        params (Dict[str, Any]): Iteration hyperparameters.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        self.logger = get_logger()
        self.p = copy.deepcopy(params)

        @staticmethod
        def _scalar(value: Any, default: Any) -> Any:
            if value is None:
                return default
            if isinstance(value, (list, tuple)):
                return value[0]
            return value

        self.p["algorithms"]  = _scalar(self.p.get("algorithms"), "xgboost")
        self.p["lr"]          = float(_scalar(self.p.get("lr"), 0.001))
        self.p["offsets"]     = int(_scalar(self.p.get("offsets"), 1))
        self.p["contextos"]   = int(_scalar(self.p.get("contextos"), 30))
        self.p["steps"]       = int(_scalar(self.p.get("steps"), 3))
        self.p["split"]       = _scalar(self.p.get("split"), "Junio")
        self.p["ano_test"]    = int(_scalar(self.p.get("ano_test"), 2014))

        self.p["salida"]                  = _scalar(self.p.get("salida"), "Qe")
        self.p["batch_size"]              = int(_scalar(self.p.get("batch_size"), 64))
        self.p["max_epochs"]              = int(_scalar(self.p.get("max_epochs"), 100))
        if isinstance(self.p.get("max_epochs_clasificador"), (list, tuple)):
            self.p["max_epochs_clasificador"] = self.p["max_epochs_clasificador"][0]
        self.p["max_epochs_clasificador"] = int(_scalar(self.p.get("max_epochs_clasificador"), 500))
        self.p["coef_de_pond"]            = float(_scalar(self.p.get("coef_de_pond"), 0.5))
        self.p["umbrales"]                = int(_scalar(self.p.get("umbrales"), 0))
        self.p["dropout"]                 = bool(_scalar(self.p.get("dropout"), False))
        self.p["l2_options"]              = bool(_scalar(self.p.get("l2_options"), False))
        self.p["use_pretrain"]    = bool(_scalar(self.p.get("use_pretrain"), True))
        self.p["eval_threshold"]  = str(_scalar(self.p.get("eval_threshold"), "p90"))
        # Ablation study: symmetric/asymmetric architecture and enabling of the
        # Random Forest meta-learner. The defaults reproduce the original
        # behavior (64-unit encoder/decoder, RF enabled).
        self.p["encoder_units"]   = int(_scalar(self.p.get("encoder_units"), 64))
        self.p["decoder_units"]   = int(_scalar(self.p.get("decoder_units"), 64))
        self.p["use_rf"]          = bool(_scalar(self.p.get("use_rf"), True))

        # Early stopping is opt-in: None disables it and always runs max_epochs.
        _esp = _scalar(self.p.get("early_stopping_patience"), None)
        self.p["early_stopping_patience"]  = int(_esp) if _esp is not None else None
        self.p["early_stopping_min_delta"] = float(_scalar(self.p.get("early_stopping_min_delta"), 1e-4))

        loss_names_cfg = self.p.get("loss_name")
        if isinstance(loss_names_cfg, (list, tuple)):
            loss_names_for_hash = sorted(set(map(str, loss_names_cfg)))
        else:
            loss_names_for_hash = [str(self.p["loss_name"])]

        hash_payload = {
            "algorithms": self.p["algorithms"],
            "lr":         self.p["lr"],
            "salida":     self.p["salida"],
            "contextos":  self.p["contextos"],
            "offsets":    self.p["offsets"],
            "steps":      self.p["steps"],
            "dropout":    self.p["dropout"],
            "l2_options": self.p["l2_options"],
            "encoder_units": self.p["encoder_units"],
            "decoder_units": self.p["decoder_units"],
            "batch_size":              self.p["batch_size"],
            "max_epochs":              self.p["max_epochs"],
            "max_epochs_clasificador": self.p["max_epochs_clasificador"],
            "early_stopping_patience":  self.p["early_stopping_patience"],
            "early_stopping_min_delta": self.p["early_stopping_min_delta"],
            "coef_de_pond": self.p["coef_de_pond"],
            "umbrales":     self.p["umbrales"],
            "loss_names" : loss_names_for_hash,
            "split":    self.p["split"],
            "ano_test": self.p["ano_test"],
            "pretrain": self.p.get("use_pretrain", True),
            "umbral": self.p.get("eval_threshold", "p90"),
        }
        # The seed enters the hash ONLY when it differs from the published
        # final run (_PUBLISHED_RUN_SEED): already-validated weights keep their
        # hash byte for byte, and a sensitivity experiment with another seed
        # trains in its own directory instead of silently resuming (which would
        # return seed-92 metrics labelled with another seed) or overwriting the
        # .keras files of the published run.
        if int(self.p.get("seed", _PUBLISHED_RUN_SEED)) != _PUBLISHED_RUN_SEED:
            hash_payload["seed"] = int(self.p["seed"])
        self.hash_id = sha1_of_dict(hash_payload)[:16]

        result_path = self.p.get("result_path", CONFIG.paths["result_path"])
        pr = str(self.p.get("numero_prueba", "53"))
        root = os.path.join(result_path,f"prueba{pr}", f"tests{pr}", self.hash_id)
        self.paths = IterationPaths(
            root=root,
            state=os.path.join(root, "iteration_state.json"),
            data=os.path.join(root, "data"),
            models=os.path.join(root, "models_saved"),
            pretrain=os.path.join(root, "pretrain_qe0"),
            ensemble=os.path.join(root, "ensemble"),
            predictions_cache=os.path.join(root, "predictions_cache"),
        )
        ensure_dirs(self.paths)

        self.state = IterationState(params=self.p)
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._norm_params: Optional[Dict] = None
        self._manifest: Optional[Dict] = None

    def _normalize_loss_names(self) -> List[str]:
        """Return the configured loss names as a list of strings.

        Returns:
            List[str]: Loss names from ``loss_name`` (or ``loss``).

        Raises:
            ValueError: If neither ``loss_name`` nor ``loss`` is configured.
        """
        loss_cfg = self.p.get("loss_name", None)
        if loss_cfg is None:
            loss_cfg = self.p.get("loss", None)
        if loss_cfg is None:
            raise ValueError("[CONFIG] Missing 'loss' or 'loss_name' in the config.")

        if isinstance(loss_cfg, (list, tuple)):
            out = []
            for l in loss_cfg:
                if isinstance(l, str):
                    out.append(l)
                else:
                    out.append(getattr(l, "__name__", str(l)))
            return out
        else:
            return [loss_cfg if isinstance(loss_cfg, str) else getattr(loss_cfg, "__name__", str(loss_cfg))]

    def _expand_valid_grid(self) -> List[Dict[str, Any]]:
        """Expand the (loss, penalty, k_high) grid, keeping valid combinations only.

        ``original_mae`` is kept only with penalty 0; the pinball losses only
        with non-zero penalties (penalty 0 would degenerate to the symmetric
        case already covered by ``original_mae``).

        Returns:
            List[Dict[str, Any]]: Unique entries with ``loss_name``, ``penalty``,
            ``k`` and ``loss_tag``.
        """
        penalties_cfg = self.p.get("pretrain_penalties", self.p.get("penalty", [0,1,2,3,4]))
        if not isinstance(penalties_cfg, (list, tuple)):
            penalties = [int(penalties_cfg)]
        else:
            penalties = [int(p) for p in penalties_cfg]

        k_list_cfg = self.p.get("k_high", [None])
        if not isinstance(k_list_cfg, (list, tuple)):
            k_list = [k_list_cfg]
        else:
            k_list = list(k_list_cfg)

        loss_names = self._normalize_loss_names()

        grid = []
        for ln in loss_names:
            ln_str = str(ln)

            if ln_str == "original_mae":
                for pen in penalties:
                    if pen == 0:
                        k_val = None
                        loss_tag = _make_loss_tag_simple(ln_str, k_val)
                        grid.append({"loss_name": ln_str, "penalty": pen, "k": k_val, "loss_tag": loss_tag})

            elif ln_str == "pinball_from_penalty":
                for pen in penalties:
                    if pen != 0:
                        k_val = None
                        loss_tag = _make_loss_tag_simple(ln_str, k_val)
                        grid.append({"loss_name": ln_str, "penalty": pen, "k": k_val, "loss_tag": loss_tag})

            elif ln_str == "pinball_weighted_from_penalty":
                for pen in penalties:
                    if pen == 0:
                        continue
                    for k_val in k_list:
                        try:
                            k_float = float(k_val)
                        except (TypeError, ValueError):
                            continue
                        if k_float > 0.0:
                            loss_tag = _make_loss_tag_simple(ln_str, k_float)
                            grid.append({"loss_name": ln_str, "penalty": pen, "k": k_float, "loss_tag": loss_tag})
                        else:
                            if "pinball_from_penalty" in loss_names:
                                loss_tag = _make_loss_tag_simple("pinball_from_penalty", None)
                                grid.append({"loss_name": "pinball_from_penalty", "penalty": pen, "k": None, "loss_tag": loss_tag})
            else:
                self.logger.warning(f"[GRID] Loss '{ln_str}' is not covered by the grid rules; ignored.")

        unique = {}
        for d in grid:
            key = (d["loss_name"], d["penalty"], d["k"])
            unique[key] = d
        return list(unique.values())

    def _denorm_qe(self, arr: np.ndarray) -> np.ndarray:
        """Undo the min-max normalization of ``Qe``.

        Args:
            arr (np.ndarray): Normalized inflow values.

        Returns:
            np.ndarray: Inflow in physical units (m3/s).

        Raises:
            RuntimeError: If the ``Qe`` normalization parameters are missing.
        """
        p = self._norm_params.get("Qe", None)
        if not p:
            raise RuntimeError("[PRED] Normalization parameters for Qe not found.")
        return arr * float(p["range"]) + float(p["min"])

    def get_train_threshold(self) -> float:
        """Extreme-event threshold fixed on the TRAINING partition.

        Reads ``self._norm_params["Qe"][eval_threshold]`` (``"p90"`` by
        default): the same immutable value used by the metric and prediction
        stages, so that no statistic is ever computed on the TEST partition.
        Requires :meth:`ensure_and_load_data` to have been called.

        Returns:
            float: The threshold in physical units.

        Raises:
            RuntimeError: If the threshold key is not in the normalization parameters.
        """
        raw_thr = self.p.get("eval_threshold", "p90")
        threshold_key = raw_thr[0] if isinstance(raw_thr, list) else raw_thr
        if self._norm_params and threshold_key in self._norm_params.get("Qe", {}):
            return float(self._norm_params["Qe"][threshold_key])
        raise RuntimeError(
            f"[TRAIN THRESHOLD] '{threshold_key}' not found in _norm_params['Qe']; "
            f"was ensure_and_load_data() called first?"
        )

    def _predict_iterative(self, encoder_model: tf.keras.Model, decoder_model: tf.keras.Model, x_enc: np.ndarray, x_dec: np.ndarray, projector_model: Optional[tf.keras.Model] = None) -> np.ndarray:
        """Generate exactly ``offsets + 1`` autoregressive predictions per window.

        One prediction per step of the target sequence
        ``[Qe(t+1), ..., Qe(t+offsets+1)]``, consuming the precomputed rows of
        ``x_dec`` (real daily rainfall forecasts). Callers read index
        ``offsets``, the LAST step, as the evaluated horizon. Generation stops
        at ``offsets + 1``: no consumer reads beyond that index.

        Args:
            encoder_model (tf.keras.Model): Encoder.
            decoder_model (tf.keras.Model): Single-step decoder.
            x_enc (np.ndarray): Encoder windows.
            x_dec (np.ndarray): Decoder inputs.
            projector_model (Optional[tf.keras.Model]): State projector for
                asymmetric encoder/decoder widths.

        Returns:
            np.ndarray: Predictions of shape ``(N, offsets + 1, 1)``.
        """
        x_enc_tf = tf.cast(x_enc, tf.float32)
        x_dec_tf = tf.cast(x_dec, tf.float32)
        state_h, state_c = encoder_model(x_enc_tf, training=False)
        if projector_model is not None:
            state_h, state_c = projector_model([state_h, state_c], training=False)
        current_input = x_dec_tf[:, :1, :]
        preds = []
        total_steps = int(self.p["offsets"]) + 1
        for step in range(total_steps):
            pred, state_h, state_c = decoder_model([current_input, state_h, state_c], training=False)
            preds.append(pred)
            if step < x_dec_tf.shape[1] - 1:
                current_input = x_dec_tf[:, step + 1:step + 2, :]
            # Unconditional (not in an else): the Qe channel must always carry
            # the previous step's prediction, exactly as in _unroll_decoder
            # (models/training.py); otherwise inference diverges from training.
            current_input = tf.concat(
                [tf.cast(pred, tf.float32), tf.cast(current_input[:, :, 1:], tf.float32)],
                axis=-1,
            )
        return tf.concat(preds, axis=1).numpy()

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Overall (hydroeval NSE/KGE) and extreme-event metrics of one series.

        Args:
            y_true (np.ndarray): Observed inflow.
            y_pred (np.ndarray): Predicted inflow.

        Returns:
            Dict[str, Dict[str, Any]]: ``overall`` (if hydroeval is installed)
            and ``top10`` blocks.

        Raises:
            RuntimeError: If the training threshold is not precomputed.
        """
        m = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_t = y_true[m].ravel()
        y_p = y_pred[m].ravel()

        out = {}
        if he is not None:
            try:
                ns  = he.nse(y_p, y_t)
                kge = he.kge(y_p, y_t)[0][0]
                out["overall"] = {"NSE": float(ns), "KGE": float(kge)}
            except Exception as e:
                self.logger.warning(f"[METRICS] hydroeval failed (overall): {e}")
        raw_thr = self.p.get("eval_threshold", "p90")
        threshold_key = raw_thr[0] if isinstance(raw_thr, list) else raw_thr
        # Top 10% of observed values.
        if y_t.size:
            # Anti-leakage: read the immutable training p90. Falling back to
            # np.quantile(y_true, ...) on TEST is forbidden: a missing training
            # threshold is a configuration error and must fail loudly.
            if not self._norm_params or "Qe" not in self._norm_params or threshold_key not in self._norm_params["Qe"]:
                raise RuntimeError(
                    f"[ANTI-LEAKAGE] Data leakage prevented: threshold {threshold_key} is not "
                    f"precomputed in norm_params. Computing quantiles on y_true during evaluation is forbidden."
                )
            thr = float(self._norm_params["Qe"][threshold_key])
            n_extreme = int(np.sum(y_t >= thr))

            try:
                # NSE and KGE on a truncated tail (y_t >= p90) are statistically
                # meaningless: both depend on the mean and covariance of the
                # COMPLETE sample. The extreme-event block therefore reports only
                # detection metrics (HitRatio/FARate/FARatio) and peak timing
                # error; NSE and KGE live exclusively in out["overall"].
                extreme = compute_extreme_metrics(y_t, y_p, threshold_p90=thr)
                timing = peak_timing_error(y_t, y_p, threshold_p90=thr)
                out["top10"] = {
                    "HitRatio": extreme["HitRatio"],
                    "FARate": extreme["FARate"],
                    "FARatio": extreme["FARatio"],
                    "PeakTimingError": timing["mean_absolute_lag"],
                    "threshold": thr, "label": "Top 10% observed", "n": n_extreme,
                }
            except Exception as e:
                self.logger.warning(f"[METRICS] Extreme-event metrics (HitRatio/FARate/FARatio/PeakTimingError) failed: {e}")
        return out

    def _check_manifest_hashes(self) -> None:
        """Log whether the bundle manifest carries file hashes.

        Only the presence of hashes is checked; the expected targets are
        listed but not compared against the files on disk.
        """
        if not self._manifest or "hashes" not in self._manifest:
            self.logger.info("[MANIFEST] No hashes found to verify.")
            return

        hashes = self._manifest["hashes"]
        targets = {
            "normalized/train.csv": os.path.join(self.paths.data, "train.csv"),
            "normalized/test.csv":  os.path.join(self.paths.data, "test.csv"),
            "normalization_params.json": os.path.join(self.paths.data, "normalization_params.json"),
        }

    def _windows_cache_paths(self) -> Tuple[str, str]:
        """Paths of the window cache, creating its directory.

        Returns:
            Tuple[str, str]: ``(windows.npz, meta.json)`` paths.
        """
        cache_dir = os.path.join(self.paths.data, "windows_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return (
            os.path.join(cache_dir, "windows.npz"),
            os.path.join(cache_dir, "meta.json"),
        )

    def _current_windows_signature(self, historicos: List[str], predicciones: List[str]) -> Dict[str, Any]:
        """Signature that invalidates the window cache when any input changes.

        Args:
            historicos (List[str]): Encoder feature columns.
            predicciones (List[str]): Rainfall forecast columns.

        Returns:
            Dict[str, Any]: Context length, offset, columns and data hashes.
        """
        man = self._manifest or {}
        h = (man.get("hashes") or {})
        return {
            "contextos": int(self.p["contextos"]),
            "offset": int(self.p["offsets"]),
            "historicos": list(historicos),
            "predicciones": list(predicciones),
            "normalized_train_csv": h.get("normalized/train.csv"),
            "normalized_test_csv": h.get("normalized/test.csv"),
        }

    def _load_windows_cache(self, historicos: List[str], predicciones: List[str]) -> bool:
        """Load cached windows into ``train_inputs``/``test_inputs`` if the signature matches.

        Args:
            historicos (List[str]): Encoder feature columns.
            predicciones (List[str]): Rainfall forecast columns.

        Returns:
            bool: ``True`` if the cache was valid and loaded.
        """
        npz_path, meta_path = self._windows_cache_paths()
        try:
            if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
                return False
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            sig_now = self._current_windows_signature(historicos, predicciones)
            if meta.get("signature") != sig_now:
                self.logger.info("[WIN/CACHE] Signature changed: windows will be recomputed.")
                return False
            data = np.load(npz_path, allow_pickle=True)
            self.train_inputs = (
                data["x_train_enc"], data["x_train_dec"], data["y_train"], data["y_train_fechas"].tolist()
            )
            self.test_inputs = (
                data["x_test_enc"], data["x_test_dec"], data["y_test"], data["y_test_fechas"].tolist()
            )
            self.p["y_test_fechas"] = data["y_test_fechas"].tolist()
            self.logger.info("[WIN/CACHE] Windows loaded from cache.")
            return True
        except Exception as e:
            self.logger.warning(f"[WIN/CACHE] Failed to load the cache: {e}")
            return False

    def _save_windows_cache(self, historicos: List[str], predicciones: List[str]) -> None:
        """Persist ``train_inputs``/``test_inputs`` and their signature.

        Args:
            historicos (List[str]): Encoder feature columns.
            predicciones (List[str]): Rainfall forecast columns.
        """
        npz_path, meta_path = self._windows_cache_paths()
        (x_tr_enc, x_tr_dec, y_tr, y_tr_fechas) = self.train_inputs
        (x_te_enc, x_te_dec, y_te, y_te_fechas) = self.test_inputs

        y_tr_fechas_arr = np.array([str(f) for f in (y_tr_fechas or [])], dtype=object)
        y_te_fechas_arr = np.array([str(f) for f in (y_te_fechas or [])], dtype=object)

        np.savez_compressed(
            npz_path,
            x_train_enc=x_tr_enc,
            x_train_dec=x_tr_dec,
            y_train=y_tr,
            y_train_fechas=y_tr_fechas_arr,
            x_test_enc=x_te_enc,
            x_test_dec=x_te_dec,
            y_test=y_te,
            y_test_fechas=y_te_fechas_arr,
        )

        meta = {
            "signature": self._current_windows_signature(historicos, predicciones),
            "shapes": {
                "x_train_enc": tuple(x_tr_enc.shape),
                "x_train_dec": tuple(x_tr_dec.shape),
                "y_train": tuple(y_tr.shape),
                "x_test_enc": tuple(x_te_enc.shape),
                "x_test_dec": tuple(x_te_dec.shape),
                "y_test": tuple(y_te.shape),
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self.logger.info("[WIN/CACHE] Windows saved to cache: %s", npz_path)

    def ensure_and_load_data(self) -> None:
        """Materialize the iteration's data bundle and load it into memory."""
        split = self.p["split"]  # split label (e.g. "Junio", "JunioExpanding")
        year  = int(self.p["ano_test"])
        self.logger.info(f"[ITER] split={split} | test_year={year} | hash={self.hash_id}")

        self.state.params["bundle_base"] = resolve_bundle_base(CONFIG, split, year)
        ensure_iteration_dataset(CONFIG, split, year, self.paths.data, self.logger)

        train_df, test_df, norm, manifest = load_iteration_bundle(self.paths.data)
        self._train_df, self._test_df, self._norm_params, self._manifest = train_df, test_df, norm, manifest
        self.state.ready_data = True
        self._check_manifest_hashes()
        self._persist_state()

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Return the loaded partitions and normalization parameters.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]: ``(train_df, test_df, norm_params)``.

        Raises:
            RuntimeError: If the data has not been loaded.
        """
        if not self.state.ready_data or self._train_df is None:
            raise RuntimeError("Data is not ready. Call ensure_and_load_data() first.")
        return self._train_df, self._test_df, self._norm_params

    def get_manifest(self) -> Dict:
        """Return the bundle manifest.

        Returns:
            Dict: The manifest, or an empty dict if not loaded.
        """
        return self._manifest or {}

    def _persist_state(self) -> None:
        """Write the iteration state (params, paths, manifest summary) to JSON."""
        manifest_summary = {}
        if self._manifest:
            for k in ["seed", "split", "test_year", "train_years", "target_col", "columns_order"]:
                if k in self._manifest:
                    manifest_summary[k] = self._manifest[k]

        with open(self.paths.state, "w", encoding="utf-8") as f:
            json.dump({
                "params": self.state.params,
                "ready_data": self.state.ready_data,
                "paths": self.paths.__dict__,
                "manifest": manifest_summary,
            }, f, ensure_ascii=False, indent=2)

    def build_windows_normal(self) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[Any]]], Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[Any]]]]:
        """Build (or load) the windows with real antecedent inflow.

        Returns:
            Tuple: ``(train_inputs, test_inputs)``, each ``(x_enc, x_dec, y, dates)``.
        """
        return self._build_windows_from_dfs(self._train_df, self._test_df, tag="normal", zero_qe=False)

    def build_windows_qe_input0(self) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[Any]]], Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[Any]]]]:
        """Build (or load) the pretraining windows with ``Qe`` inputs masked to zero.

        Returns:
            Tuple: ``(train_inputs, test_inputs)``, each ``(x_enc, x_dec, y, dates)``.
        """
        return self._build_windows_from_dfs(self._train_df, self._test_df, tag="qe_input0", zero_qe=True)

    def _build_windows_from_dfs(self, train_df: pd.DataFrame, test_df: pd.DataFrame, tag: str, zero_qe: bool) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[Any]]], Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[Any]]]]:
        """Build encoder/decoder windows for train and test, with a per-tag cache.

        Args:
            train_df (pd.DataFrame): Normalized training partition.
            test_df (pd.DataFrame): Normalized test partition.
            tag (str): Cache tag (``normal`` or ``qe_input0``).
            zero_qe (bool): Mask the ``Qe`` channel of encoder and decoder
                inputs to zero (rainfall-only pretraining).

        Returns:
            Tuple: ``(train_inputs, test_inputs)``, each ``(x_enc, x_dec, y, dates)``.

        Raises:
            RuntimeError: If the data or the manifest column order is missing.
            ValueError: If a required column is missing.
            AssertionError: If the window shapes are inconsistent.
        """
        if not self.state.ready_data:
            raise RuntimeError("Data is not ready. Call ensure_and_load_data() first.")
        if not self._manifest or "columns_order" not in self._manifest:
            raise RuntimeError("Manifest without 'columns_order'.")

        columns_order   = list(self._manifest["columns_order"])
        variable_salida = self._manifest.get("target_col", "Qe")
        historicos      = [c for c in columns_order if not c.startswith("pred")]
        predicciones    = [c for c in columns_order if c.startswith("pred")]

        for df_name, df in [("train", train_df), ("test", test_df)]:
            missing = [c for c in historicos + predicciones + [variable_salida, "Fecha"] if c not in df.columns]
            if missing:
                raise ValueError(f"[build_windows/{tag}] {df_name} missing columns: {missing}")

        contextos = int(self.p["contextos"])
        offset    = int(self.p["offsets"])

        orig_cache_fn = self._windows_cache_paths

        def _tagged_cache_paths() -> Tuple[str, str]:
            cache_dir = os.path.join(self.paths.data, f"windows_cache_{tag}")
            os.makedirs(cache_dir, exist_ok=True)
            return (
                os.path.join(cache_dir, "windows.npz"),
                os.path.join(cache_dir, "meta.json"),
            )
        self._windows_cache_paths = _tagged_cache_paths

        if self._load_windows_cache(historicos, predicciones):
            self._persist_state()
            self.logger.info("[WIN/%s] Windows (cache) train enc=%s dec=%s y=%s | test enc=%s dec=%s y=%s",
                            tag,
                            tuple(self.train_inputs[0].shape), tuple(self.train_inputs[1].shape), tuple(self.train_inputs[2].shape),
                            tuple(self.test_inputs[0].shape),  tuple(self.test_inputs[1].shape),  tuple(self.test_inputs[2].shape))
            self._windows_cache_paths = orig_cache_fn
            return self.train_inputs, self.test_inputs

        (x_tr_enc, x_tr_dec, y_tr, y_tr_fechas), (_,_,_,_), (x_te_enc, x_te_dec, y_te, y_te_fechas) = prepare_model_inputs(
            train_norm=train_df, val_norm=None, test_norm=test_df,
            variable_salida=variable_salida,
            historicos=historicos, predicciones=predicciones,
            contextos=contextos, offset=offset
        )

        if zero_qe:
            try:
                qe_idx = historicos.index(variable_salida)
            except ValueError:
                raise ValueError(f"[build_windows/{tag}] {variable_salida} not found in historicos.")
            x_tr_enc[:, :, qe_idx] = 0.0
            x_te_enc[:, :, qe_idx] = 0.0
            x_tr_dec[:, :, 0] = 0.0
            x_te_dec[:, :, 0] = 0.0

        enc_feat_expected = len(historicos)
        assert isinstance(x_tr_enc, np.ndarray) and x_tr_enc.ndim == 3 and x_tr_enc.shape[1] == contextos and x_tr_enc.shape[2] == enc_feat_expected
        assert isinstance(x_te_enc, np.ndarray) and x_te_enc.ndim == 3 and x_te_enc.shape[1] == contextos and x_te_enc.shape[2] == enc_feat_expected

        # sliding_window() produces exactly offset + 1 decoder and target steps,
        # one per day [t+1..t+offset+1].
        expected_dec_len = offset + 1

        assert x_tr_dec.ndim == 3 and x_tr_dec.shape[1] == expected_dec_len and x_tr_dec.shape[2] == 5
        assert x_te_dec.ndim == 3 and x_te_dec.shape[1] == expected_dec_len and x_te_dec.shape[2] == 5

        assert y_tr.ndim == 2 and y_tr.shape[1] == expected_dec_len
        assert y_te.ndim == 2 and y_te.shape[1] == expected_dec_len
        if y_tr_fechas is not None: assert len(y_tr_fechas) == y_tr.shape[0]
        if y_te_fechas is not None: assert len(y_te_fechas) == y_te.shape[0]

        self.train_inputs = (x_tr_enc, x_tr_dec, y_tr, list(y_tr_fechas) if y_tr_fechas is not None else None)
        self.test_inputs  = (x_te_enc, x_te_dec, y_te, list(y_te_fechas) if y_te_fechas is not None else None)
        self.p["y_test_fechas"] = list(map(str, y_te_fechas)) if y_te_fechas is not None else None

        self._save_windows_cache(historicos, predicciones)
        self._persist_state()

        self.logger.info("[WIN/%s] Windows built: train enc=%s dec=%s y=%s | test enc=%s dec=%s y=%s",
                        tag,
                        tuple(x_tr_enc.shape), tuple(x_tr_dec.shape), tuple(y_tr.shape),
                        tuple(x_te_enc.shape), tuple(x_te_dec.shape), tuple(y_te.shape))
        self._windows_cache_paths = orig_cache_fn
        self._windows_tag = tag
        self._persist_state()
        return self.train_inputs, self.test_inputs

    def ensure_pretrained(self) -> None:
        """Pretrain one model per grid entry on ``Qe``-masked windows, if missing.

        Raises:
            ValueError: If the valid grid is empty.
        """
        os.makedirs(self.paths.pretrain, exist_ok=True)

        grid = self._expand_valid_grid()
        if not grid:
            raise ValueError("[PRETRAIN] The grid of valid combinations is empty (check config and rules).")

        encoder_units = int(self.p.get("encoder_units", 64))
        decoder_units = int(self.p.get("decoder_units", 64))
        needs_projector = encoder_units != decoder_units

        expected = []
        for g in grid:
            enc = os.path.join(self.paths.pretrain, f"encoder_{g['loss_tag']}_{g['penalty']}.keras")
            dec = os.path.join(self.paths.pretrain, f"decoder_{g['loss_tag']}_{g['penalty']}.keras")
            expected.extend([enc, dec])
            if needs_projector:
                expected.append(os.path.join(self.paths.pretrain, f"projector_{g['loss_tag']}_{g['penalty']}.keras"))

        missing = [p for p in expected if not os.path.exists(p)]
        if not missing:
            self.logger.info("[PRETRAIN] %d models already present in %s. Skipping.", len(expected)//2, self.paths.pretrain)
            return

        (x_tr_enc, x_tr_dec, y_tr, _), _ = self.build_windows_qe_input0()
        self.logger.info("[PRETRAIN] Valid grid (|combos|=%d): %s", len(grid), grid)

        contextos = int(self.p["contextos"])
        n_enc_feat = x_tr_enc.shape[2]
        n_dec_feat = x_tr_dec.shape[2]

        for g in grid:
            loss_tag = g["loss_tag"]
            pen = g["penalty"]
            k_val = g["k"]

            enc_path = os.path.join(self.paths.pretrain, f"encoder_{loss_tag}_{pen}.keras")
            dec_path = os.path.join(self.paths.pretrain, f"decoder_{loss_tag}_{pen}.keras")
            proj_path = os.path.join(self.paths.pretrain, f"projector_{loss_tag}_{pen}.keras")

            it_p = dict(self.p)
            it_p.update({
                "loss_name": g["loss_name"],
                "penalty": int(pen),
                "result_folder": self.paths.pretrain,
                "batch_size": int(_scalar(it_p.get("batch_size"), 64)),
                "max_epochs": int(_scalar(it_p.get("max_epochs_pretrain"), it_p.get("max_epochs", 50))),
                "lr": float(_scalar(it_p.get("lr"), 0.001)),
                "dropout": bool(_scalar(it_p.get("dropout"), False)),
                "l2_options": bool(_scalar(it_p.get("l2_options"), False)),
                "test_year": int(self.p["ano_test"]),
            })
            if g["loss_name"] == "pinball_weighted_from_penalty" and (k_val is not None):
                it_p["k_high"] = float(k_val)
            else:
                it_p.pop("k_high", None)

            encoder_model, _, _, _ = build_encoder((contextos, n_enc_feat), it_p)
            decoder_model = build_decoder(n_dec_feat, it_p)
            projector_model = build_state_projector(encoder_units, decoder_units)

            self.logger.info("[PRETRAIN] Training loss=%s | pen=%s | k_high=%s…", g["loss_name"], pen, it_p.get("k_high"))
            _ = train_model(encoder_model, decoder_model, x_tr_enc, x_tr_dec, y_tr, None, None, None, it_p, projector_model=projector_model)
            encoder_model.save(enc_path)
            decoder_model.save(dec_path)
            if projector_model is not None:
                projector_model.save(proj_path)
            self.logger.info("[PRETRAIN] Saved: %s | %s%s", enc_path, dec_path, f" | {proj_path}" if projector_model is not None else "")

        self.logger.info("[PRETRAIN] Done. Folder: %s", self.paths.pretrain)

    def get_model_groups(self) -> Dict[str, List[str]]:
        """List the saved ``.keras`` models by stage.

        Returns:
            Dict[str, List[str]]: ``pretrained`` and ``finetuned`` file names.
        """
        def _scan(dirpath: str) -> List[str]:
            if not os.path.exists(dirpath):
                return []
            return sorted([f for f in os.listdir(dirpath) if f.endswith(".keras")])
        return {
            "pretrained": _scan(self.paths.pretrain),
            "finetuned":  _scan(self.paths.models),
    }

    def finetune(self) -> None:
        """Fine-tune (or train from scratch) one model per grid entry, if missing.

        With ``use_pretrain=True`` each model resumes from its pretrained
        checkpoint (running :meth:`ensure_pretrained` if needed). With
        ``use_pretrain=False`` encoder and decoder are built with fresh random
        weights directly on the training windows, without touching the
        pretraining folder; this is what distinguishes, e.g., M1 from M3.

        Raises:
            FileNotFoundError: If pretrained models are still missing after
                :meth:`ensure_pretrained`.
        """
        os.makedirs(self.paths.models, exist_ok=True)

        if not (hasattr(self, "train_inputs") and hasattr(self, "test_inputs") and getattr(self, "_windows_tag", None) == "normal"):
            (x_tr_enc, x_tr_dec, y_tr, _), _ = self.build_windows_normal()
        else:
            x_tr_enc, x_tr_dec, y_tr, _ = self.train_inputs

        grid = self._expand_valid_grid()
        self.logger.info("[FINETUNE] Valid grid (|combos|=%d): %s", len(grid), grid)

        use_pretrain = bool(self.p.get("use_pretrain", True))
        contextos = int(self.p["contextos"])
        n_enc_feat = x_tr_enc.shape[2]
        n_dec_feat = x_tr_dec.shape[2]
        encoder_units = int(self.p.get("encoder_units", 64))
        decoder_units = int(self.p.get("decoder_units", 64))
        needs_projector = encoder_units != decoder_units

        for g in grid:
            loss_tag = g["loss_tag"]
            pen = g["penalty"]
            k_val = g["k"]

            fin_enc = os.path.join(self.paths.models,  f"encoder_{loss_tag}_{pen}.keras")
            fin_dec = os.path.join(self.paths.models,  f"decoder_{loss_tag}_{pen}.keras")
            fin_proj = os.path.join(self.paths.models,  f"projector_{loss_tag}_{pen}.keras")

            if os.path.exists(fin_enc) and os.path.exists(fin_dec) and (not needs_projector or os.path.exists(fin_proj)):
                self.logger.info("[FINETUNE] Skip %s pen=%s (already present in %s)", loss_tag, pen, self.paths.models)
                continue

            it_p = dict(self.p)
            it_p.update({
                "penalty": int(pen),
                "loss_name": g["loss_name"],
                "result_folder": self.paths.models,
                "batch_size": int(_scalar(self.p.get("batch_size"), 64)),
                "max_epochs": int(_scalar(self.p.get("max_epochs"), 50)),
                "lr": float(_scalar(self.p.get("lr"), 0.001)),
                "test_year": int(self.p["ano_test"]),
            })
            if g["loss_name"] == "pinball_weighted_from_penalty" and (k_val is not None):
                it_p["k_high"] = float(k_val)
            else:
                it_p.pop("k_high", None)

            if use_pretrain:
                pre_enc = os.path.join(self.paths.pretrain, f"encoder_{loss_tag}_{pen}.keras")
                pre_dec = os.path.join(self.paths.pretrain, f"decoder_{loss_tag}_{pen}.keras")
                pre_proj = os.path.join(self.paths.pretrain, f"projector_{loss_tag}_{pen}.keras")
                if not (os.path.exists(pre_enc) and os.path.exists(pre_dec) and (not needs_projector or os.path.exists(pre_proj))):
                    self.logger.info("[FINETUNE] Pretrained models missing for %s. Calling ensure_pretrained()…", loss_tag)
                    self.ensure_pretrained()
                    if not (os.path.exists(pre_enc) and os.path.exists(pre_dec) and (not needs_projector or os.path.exists(pre_proj))):
                        raise FileNotFoundError(f"[FINETUNE] Pretrained models not found: {pre_enc} / {pre_dec}"
                                                 f"{' / ' + pre_proj if needs_projector else ''}")
                self.logger.info("[FINETUNE] Loading pretrained models: %s | %s", pre_enc, pre_dec)
                encoder_model = tf.keras.models.load_model(pre_enc)
                decoder_model = tf.keras.models.load_model(pre_dec)
                projector_model = tf.keras.models.load_model(pre_proj) if needs_projector else None
                self.logger.info("[FINETUNE] Training (from pretrained) loss=%s | pen=%s | k_high=%s…",
                                  g["loss_name"], pen, it_p.get("k_high"))
            else:
                encoder_model, _, _, _ = build_encoder((contextos, n_enc_feat), it_p)
                decoder_model = build_decoder(n_dec_feat, it_p)
                projector_model = build_state_projector(encoder_units, decoder_units)
                self.logger.info("[FINETUNE] use_pretrain=False -> training from scratch (paths.pretrain "
                                  "is not touched) loss=%s | pen=%s | k_high=%s…", g["loss_name"], pen, it_p.get("k_high"))

            _ = train_model(
                encoder_model=encoder_model,
                decoder_model=decoder_model,
                x_train_encoder=x_tr_enc, x_train_decoder=x_tr_dec, y_train=y_tr,
                x_val_encoder=None, x_val_decoder=None, y_val=None,
                iteration_params=it_p,
                projector_model=projector_model,
            )
            encoder_model.save(fin_enc)
            decoder_model.save(fin_dec)
            if projector_model is not None:
                projector_model.save(fin_proj)
            self.logger.info("[FINETUNE] Saved: %s | %s%s", fin_enc, fin_dec, f" | {fin_proj}" if projector_model is not None else "")

    def ensure_predictions(self, force: bool = False, save_csv: bool = True, subset: str = "test") -> List[Dict[str, Any]]:
        """Generate (or reuse) the denormalized predictions of every fine-tuned model.

        The evaluated horizon is the LAST decoder step (index ``offsets``). A
        cached ``.npz`` that cannot be read is regenerated.

        Args:
            force (bool): Recompute predictions even if cached.
            save_csv (bool): Also write a CSV with observed/predicted values
                and the training-threshold mask.
            subset (str): ``"test"`` or ``"train"``.

        Returns:
            List[Dict[str, Any]]: One entry per model with ``penalty``,
            ``loss_name``, ``base``, ``npz`` and ``csv``.

        Raises:
            ValueError: If ``subset`` is invalid.
            RuntimeError: If the training threshold is not precomputed.
        """
        os.makedirs(self.paths.predictions_cache, exist_ok=True)

        grid = self._expand_valid_grid()
        if not grid:
            self.logger.warning("[PRED/%s] Empty valid grid. Nothing to predict.", subset)
            return []

        have_normal = (
            hasattr(self, "train_inputs") and hasattr(self, "test_inputs")
            and getattr(self, "_windows_tag", None) == "normal"
        )
        if not have_normal:
            (x_tr_enc, x_tr_dec, y_tr, tr_dates), (x_te_enc, x_te_dec, y_te, te_dates) = self.build_windows_normal()
        else:
            x_tr_enc, x_tr_dec, y_tr, tr_dates = self.train_inputs
            x_te_enc, x_te_dec, y_te, te_dates = self.test_inputs

        if subset == "test":
            x_enc, x_dec, y_lab, fechas = x_te_enc, x_te_dec, y_te, te_dates
        elif subset == "train":
            x_enc, x_dec, y_lab, fechas = x_tr_enc, x_tr_dec, y_tr, tr_dates
        else:
            raise ValueError("subset must be 'test' or 'train'")

        self.finetune()

        # y_lab is the full sequence [Qe(t+1)...Qe(t+offsets+1)]; the evaluated
        # horizon is its LAST step (t+offsets+1), matching step_idx below.
        y_true_norm = y_lab[:, -1].astype(float)
        y_true = self._denorm_qe(y_true_norm)
        fechas = pd.to_datetime(fechas) if fechas is not None else pd.date_range(start=0, periods=y_true.shape[0])

        step_idx = int(self.p.get("offsets", 1))
        encoder_units = int(self.p.get("encoder_units", 64))
        decoder_units = int(self.p.get("decoder_units", 64))
        needs_projector = encoder_units != decoder_units

        self.logger.info("[PRED/%s] Generating predictions for %d valid combinations.", subset, len(grid))
        results: list[dict] = []

        for g in grid:
            loss_tag = g["loss_tag"]
            pen = int(g["penalty"])

            base = f"{loss_tag}_{pen}" if subset == "test" else f"{loss_tag}_{pen}_{subset}"
            npz = os.path.join(self.paths.predictions_cache, f"pred_{base}.npz")
            csv = os.path.join(self.paths.predictions_cache, f"pred_{base}.csv")

            fin_enc = os.path.join(self.paths.models, f"encoder_{loss_tag}_{pen}.keras")
            fin_dec = os.path.join(self.paths.models, f"decoder_{loss_tag}_{pen}.keras")
            fin_proj = os.path.join(self.paths.models, f"projector_{loss_tag}_{pen}.keras")

            if not (os.path.exists(fin_enc) and os.path.exists(fin_dec) and (not needs_projector or os.path.exists(fin_proj))):
                self.logger.warning("[PRED/%s] Skipping %s: fine-tuned models not found (%s | %s).",
                                    subset, base, os.path.basename(fin_enc), os.path.basename(fin_dec))
                continue

            need_pred = force or not os.path.exists(npz)
            if need_pred:
                enc_model = tf.keras.models.load_model(fin_enc)
                dec_model = tf.keras.models.load_model(fin_dec)
                proj_model = tf.keras.models.load_model(fin_proj) if needs_projector else None

                preds_seq = self._predict_iterative(enc_model, dec_model, x_enc, x_dec, projector_model=proj_model)
                y_pred_norm = preds_seq[:, step_idx, 0]
                y_pred = self._denorm_qe(y_pred_norm)

                np.savez_compressed(
                    npz,
                    dates=np.array(fechas.astype('datetime64[ns]')).astype('datetime64[ns]'),
                    y_true=y_true, y_pred=y_pred,
                    y_true_norm=y_true_norm, y_pred_norm=y_pred_norm,
                    step_idx=step_idx,
                )
                raw_thr = self.p.get("eval_threshold", "p90")
                threshold_key = raw_thr[0] if isinstance(raw_thr, list) else raw_thr
                if save_csv:
                    # Anti-leakage: quantiles are never computed on TEST.
                    if not self._norm_params or "Qe" not in self._norm_params or threshold_key not in self._norm_params["Qe"]:
                        raise RuntimeError("[ANTI-LEAKAGE] Computing dynamic quantiles on Test is forbidden.")
                    thr = float(self._norm_params["Qe"][threshold_key])

                    mask_top = (y_true >= thr)
                    hit_cls = (y_pred >= y_true).astype(int)
                    df = pd.DataFrame({
                        "Date": fechas,
                        "Observed": y_true,
                        "Pred": y_pred,
                        "Band_min": np.nan,
                        "Band_max": np.nan,
                        "Classifier": y_pred,
                        "TopMask": mask_top.astype(int),
                        "Classifier_Hit": hit_cls,
                    })
                    df.to_csv(csv, index=False)

                self.logger.info("[PRED/%s] Predictions generated: %s", subset, os.path.basename(npz))
            else:
                try:
                    data = np.load(npz)
                    step_idx = int(data["step_idx"])
                except Exception as e:
                    self.logger.warning("[PRED/%s] Could not read %s (recomputing). Error: %s", subset, npz, e)
                    os.remove(npz)
                    enc_model = tf.keras.models.load_model(fin_enc)
                    dec_model = tf.keras.models.load_model(fin_dec)
                    proj_model = tf.keras.models.load_model(fin_proj) if needs_projector else None
                    preds_seq = self._predict_iterative(enc_model, dec_model, x_enc, x_dec, projector_model=proj_model)
                    y_pred_norm = preds_seq[:, step_idx, 0]
                    y_pred = self._denorm_qe(y_pred_norm)
                    np.savez_compressed(
                        npz,
                        dates=np.array(fechas.astype('datetime64[ns]')).astype('datetime64[ns]'),
                        y_true=y_true, y_pred=y_pred,
                        y_true_norm=y_true_norm, y_pred_norm=y_pred_norm,
                        step_idx=step_idx,
                    )
                    if save_csv:
                        # Anti-leakage: quantiles are never computed on TEST.
                        if not self._norm_params or "Qe" not in self._norm_params or threshold_key not in self._norm_params["Qe"]:
                            raise RuntimeError("[ANTI-LEAKAGE] Computing dynamic quantiles on Test is forbidden.")
                        thr = float(self._norm_params["Qe"][threshold_key])

                        mask_top = (y_true >= thr)
                        hit_cls = (y_pred >= y_true).astype(int)
                        df = pd.DataFrame({
                            "Date": fechas,
                            "Observed": y_true,
                            "Pred": y_pred,
                            "Band_min": np.nan,
                            "Band_max": np.nan,
                            "Classifier": y_pred,
                            "TopMask": mask_top.astype(int),
                            "Classifier_Hit": hit_cls,
                        })
                        df.to_csv(csv, index=False)
                    self.logger.info("[PRED/%s] Predictions regenerated after a failure: %s", subset, os.path.basename(npz))

            results.append({
                "penalty": pen,
                "loss_name": g["loss_name"],
                "base": base,
                "npz": npz,
                "csv": csv,
            })

        return results

    def _hydro_year_of(self, fechas: Any) -> np.ndarray:
        """Hydrological-year label (July-June cycle) of each date.

        The label is the starting year of the cycle: a date in January-June
        of year Y belongs to cycle Y-1.

        Args:
            fechas (Any): Date-like sequence.

        Returns:
            np.ndarray: Integer hydrological year per date.
        """
        idx = pd.to_datetime(pd.Index(fechas))
        return (idx.year - (idx.month < 7).astype(int)).to_numpy()

    def _oof_cache_dir(self) -> str:
        """Directory of the internal out-of-fold predictions, created if needed.

        Returns:
            str: The directory path.
        """
        d = os.path.join(self.paths.root, "oof_internal")
        os.makedirs(d, exist_ok=True)
        return d

    def _train_and_predict_internal(self, train_sub: pd.DataFrame, test_sub: pd.DataFrame, g: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, List[Any]]]:
        """Train ONE model on ``train_sub`` and predict ``test_sub``, in isolation.

        Does not touch the outer fold's data, windows or model folders. Reuses
        the same low-level pieces as the main pipeline (window construction,
        model builders, training loop and iterative prediction) on arbitrary
        sub-dataframes.

        Args:
            train_sub (pd.DataFrame): Internal training years.
            test_sub (pd.DataFrame): Internal held-out year.
            g (Dict[str, Any]): Grid entry (``loss_name``, ``penalty``, ``k``).

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray, List[Any]]]:
            ``(y_true, y_pred, dates)``, or ``None`` if either internal window
            set is empty (year too short after subtracting the context).
        """
        columns_order = list(self._manifest["columns_order"])
        variable_salida = self._manifest.get("target_col", "Qe")
        historicos = [c for c in columns_order if not c.startswith("pred")]
        predicciones = [c for c in columns_order if c.startswith("pred")]
        contextos = int(self.p["contextos"])
        offset = int(self.p["offsets"])

        (x_tr_enc, x_tr_dec, y_tr, _), _, (x_te_enc, x_te_dec, y_te, te_dates) = prepare_model_inputs(
            train_norm=train_sub, val_norm=None, test_norm=test_sub,
            variable_salida=variable_salida, historicos=historicos, predicciones=predicciones,
            contextos=contextos, offset=offset,
        )
        if x_tr_enc.shape[0] == 0 or x_te_enc.shape[0] == 0:
            return None

        encoder_units = int(self.p.get("encoder_units", 64))
        decoder_units = int(self.p.get("decoder_units", 64))
        needs_projector = encoder_units != decoder_units
        n_enc_feat, n_dec_feat = x_tr_enc.shape[2], x_tr_dec.shape[2]

        it_p = dict(self.p)
        it_p.update({"loss_name": g["loss_name"], "penalty": int(g["penalty"])})
        if g["loss_name"] == "pinball_weighted_from_penalty" and (g["k"] is not None):
            it_p["k_high"] = float(g["k"])
        else:
            it_p.pop("k_high", None)

        encoder_model, _, _, _ = build_encoder((contextos, n_enc_feat), it_p)
        decoder_model = build_decoder(n_dec_feat, it_p)
        projector_model = build_state_projector(encoder_units, decoder_units) if needs_projector else None

        train_model(encoder_model, decoder_model, x_tr_enc, x_tr_dec, y_tr,
                    None, None, None, it_p, projector_model=projector_model)

        preds_seq = self._predict_iterative(encoder_model, decoder_model, x_te_enc, x_te_dec,
                                             projector_model=projector_model)
        y_pred = self._denorm_qe(preds_seq[:, offset, 0])
        y_true = self._denorm_qe(y_te[:, -1].astype(float))
        return y_true, y_pred, te_dates

    def ensure_oof_train_predictions(self, force: bool = False) -> List[Dict[str, Any]]:
        """Nested blocked out-of-fold training predictions for the meta-learner.

        Source of :meth:`ensure_ensemble_labels` with ``subset="train"`` and of
        :meth:`train_classifier`. It never affects the outer fold's final model
        (pretraining, fine-tuning and test predictions are unchanged).

        Strategy (``self.p["cv_strategy"]``):

        - ``"expanding"`` (default): nested expanding window with a burn-in of
          ``self.p["burnin_years"]`` hydrological years; for each internal year
          ``t`` after the burn-in, train on ``[start..t-1]`` and predict ``t``.
        - ``"loyo"``: group k-fold by hydrological year, without burn-in; for
          each internal year ``k``, train on the other internal years and
          predict ``k``.

        Args:
            force (bool): Recompute even if cached.

        Returns:
            List[Dict[str, Any]]: One entry per grid model with ``loss_name``,
            ``penalty``, ``base`` and ``npz``.

        Raises:
            RuntimeError: If the grid is empty, the burn-in consumes every
                year, or no internal year yields valid windows.
            ValueError: If the CV strategy is unknown.
        """
        cache_dir = self._oof_cache_dir()
        grid = self._expand_valid_grid()
        if not grid:
            raise RuntimeError("[OOF] Empty valid grid; the classifier meta-training set cannot be built.")

        hyd_years = self._hydro_year_of(self._train_df["Fecha"])
        years_present = sorted(pd.unique(hyd_years).tolist())

        strategy = str(self.p.get("cv_strategy", "expanding"))
        if strategy == "expanding":
            burnin_years = int(self.p.get("burnin_years", CONFIG.cv["burnin_years"]))
            if len(years_present) <= burnin_years:
                raise RuntimeError(
                    f"[OOF] burnin_years={burnin_years} consumes all the available years "
                    f"in train ({years_present}) or more -- no internal year is left for "
                    f"the OOF loop. This outer fold (ano_test={self.p.get('ano_test')}) should "
                    f"not reach ensure_oof_train_predictions() under the truncated "
                    f"expanding CV; check EXPANDING_CV_YEARS."
                )
            internal_splits = [
                (years_present[idx], years_present[:idx])
                for idx in range(burnin_years, len(years_present))
            ]
        elif strategy == "loyo":
            internal_splits = [
                (t, [y for y in years_present if y != t])
                for t in years_present
            ]
        else:
            raise ValueError(f"[OOF] Unknown self.p['cv_strategy']: {strategy!r} (expected 'expanding' or 'loyo').")

        results: list[dict] = []
        for g in grid:
            base = f"{g['loss_tag']}_{int(g['penalty'])}"
            npz = os.path.join(cache_dir, f"oof_{base}.npz")

            if os.path.exists(npz) and not force:
                results.append({"loss_name": g["loss_name"], "penalty": int(g["penalty"]), "base": base, "npz": npz})
                continue

            y_true_parts, y_pred_parts, dates_parts = [], [], []
            for t, train_years_internal in internal_splits:
                train_sub = self._train_df.loc[np.isin(hyd_years, train_years_internal)].reset_index(drop=True)
                test_sub = self._train_df.loc[hyd_years == t].reset_index(drop=True)
                out = self._train_and_predict_internal(train_sub, test_sub, g)
                if out is None:
                    self.logger.warning("[OOF/%s] Internal year %s has no valid windows; skipped.", base, t)
                    continue
                yt, yp, dt = out
                y_true_parts.append(yt)
                y_pred_parts.append(yp)
                dates_parts.append(dt)

            if not y_true_parts:
                raise RuntimeError(f"[OOF/{base}] No internal year produced valid windows -- no OOF predictions to fit.")

            y_true = np.concatenate(y_true_parts)
            y_pred = np.concatenate(y_pred_parts)
            dates = pd.to_datetime(pd.Index(np.concatenate([pd.DatetimeIndex(d).values for d in dates_parts])))
            order = np.argsort(dates.values, kind="stable")
            y_true, y_pred, dates = y_true[order], y_pred[order], dates[order]

            np.savez_compressed(
                npz,
                dates=np.array(dates.astype("datetime64[ns]")),
                y_true=y_true, y_pred=y_pred,
            )
            results.append({"loss_name": g["loss_name"], "penalty": int(g["penalty"]), "base": base, "npz": npz})

        return results

    def ensure_ensemble_labels(self, subset: str = "test", force: bool = False) -> Dict[str, Any]:
        """Label each time step with the candidate model that minimizes the asymmetric loss.

        Training labels come from genuine out-of-fold predictions
        (:meth:`ensure_oof_train_predictions`), not from in-sample predictions.
        The label is a formal evaluation of the configured asymmetric loss at
        the MAXIMUM penalty of the grid (the most conservative criterion), so
        that the meta-learner learns exactly that rule. A secondary ranking
        orders models by over-prediction first, then by absolute
        under-prediction.

        Args:
            subset (str): ``"test"`` or ``"train"``.
            force (bool): Recompute even if cached.

        Returns:
            Dict[str, Any]: ``subset``, ``n``, ``npz``, ``json`` and ``csv``
            paths (plus ``ranking`` when freshly computed).

        Raises:
            RuntimeError: If no predictions are available.
            ValueError: If ``y_true`` differs across models or the data CSV
                has no ``Fecha`` column.
            FileNotFoundError: If the subset CSV is missing.
        """
        os.makedirs(self.paths.ensemble, exist_ok=True)

        lab_npz = os.path.join(self.paths.ensemble, f"ensemble_labels_{subset}.npz")
        lab_json = os.path.join(self.paths.ensemble, f"ensemble_labels_{subset}.json")
        out_csv  = os.path.join(self.paths.data, f"{subset}_labels.csv")

        if (os.path.exists(lab_npz) and os.path.exists(out_csv)) and not force:
            self.logger.info("[ENS/%s] Labels already exist. Skip.", subset)
            data = np.load(lab_npz, allow_pickle=True)
            return {
                "subset": subset,
                "n": int(data["indices"].shape[0]),
                "npz": lab_npz,
                "json": lab_json,
                "csv": out_csv,
            }

        if subset == "train":
            # Training labels of the classifier come from genuine OOF
            # predictions, not from the in-sample ones of ensure_predictions.
            pred_list = self.ensure_oof_train_predictions(force=force)
        else:
            pred_list = self.ensure_predictions(force=False, save_csv=True, subset=subset)
        if len(pred_list) == 0:
            raise RuntimeError(f"[ENS/{subset}] No predictions available.")

        y_true_ref, dates_ref = None, None
        preds_stack = []
        model_bases_raw = []

        for pr in pred_list:
            d = np.load(pr["npz"], allow_pickle=True)
            y_true = d["y_true"].astype(float)
            y_pred = d["y_pred"].astype(float)
            dates  = pd.to_datetime(d["dates"])

            if y_true_ref is None:
                y_true_ref, dates_ref = y_true, dates
            else:
                if y_true.shape != y_true_ref.shape or np.nanmax(np.abs(y_true - y_true_ref)) > 1e-9:
                    raise ValueError(f"[ENS/{subset}] Inconsistent y_true across models.")

            preds_stack.append(y_pred)
            model_bases_raw.append(pr["base"])

        predictions_all = np.vstack(preds_stack)
        model_bases = [_canonical_base(b) for b in model_bases_raw]

        num_models, N = predictions_all.shape

        diffs = predictions_all - y_true_ref[None, :]

        # Ensemble label = formal evaluation of the configured asymmetric loss,
        # not an ad-hoc heuristic. The MAXIMUM penalty of the grid
        # (self.p["penalty"]) is used as the most conservative criterion: every
        # candidate competes under the same rule (penalize inflow
        # under-prediction hardest), so the RF learns exactly that criterion.
        loss_name = self.p.get("loss_name", "original_mae")
        if isinstance(loss_name, (list, tuple)):
            loss_name = loss_name[0]
        penalty_cfg = self.p.get("penalty", 0)
        max_penalty = max(penalty_cfg) if isinstance(penalty_cfg, (list, tuple)) else penalty_cfg

        loss_values = _asymmetric_loss_elementwise(
            y_true_ref[None, :], predictions_all, str(loss_name), float(max_penalty)
        )  # (num_models, N)

        best_model_indices = np.argmin(loss_values, axis=0).astype(np.int32)
        optimal_values = predictions_all[best_model_indices, np.arange(N)]

        ranking = np.zeros((num_models, N), dtype=np.int32)

        for j in range(N):
            dif_j = diffs[:, j]
            over_idx  = np.where(dif_j >= 0)[0]
            under_idx = np.where(dif_j < 0)[0]

            if over_idx.size > 0:
                over_order = over_idx[np.argsort(dif_j[over_idx])]
            else:
                over_order = np.empty(0, dtype=int)

            if under_idx.size > 0:
                under_order = under_idx[np.argsort(np.abs(dif_j[under_idx]))]
            else:
                under_order = np.empty(0, dtype=int)

            ranking[:, j] = np.concatenate([over_order, under_order])

        np.savez_compressed(
            lab_npz,
            indices=best_model_indices,
            optimal_values=optimal_values,
            models=np.array(model_bases, dtype=object),
            dates=np.array(dates_ref.astype("datetime64[ns]")),
            ranking=ranking,
        )

        summary = {
            "subset": subset,
            "n_samples": int(best_model_indices.size),
            "n_models": num_models,
            "models_order": model_bases,
            "rule": f"argmin_{loss_name}_penalty_{max_penalty}",
            "has_ranking": True,
        }
        with open(lab_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        df_path = os.path.join(self.paths.data, f"{subset}.csv")
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"[ENS/{subset}] {df_path} not found to attach the labels.")

        df = pd.read_csv(df_path)
        if "Fecha" not in df.columns:
            raise ValueError(f"[ENS/{subset}] {df_path} has no 'Fecha' column.")

        p = self._norm_params["Qe"]
        opt_norm = (optimal_values - float(p["min"])) / float(p["range"]) if float(p["range"]) != 0 else np.zeros_like(optimal_values)

        labels_df = pd.DataFrame({
            "Fecha": pd.to_datetime(dates_ref),
            "ensemble_idx": best_model_indices,
            "ensemble_pred": optimal_values,
            "ensemble_pred_norm": opt_norm,
        })

        df["Fecha"] = pd.to_datetime(df["Fecha"])
        df = df.drop_duplicates(subset=["Fecha"], keep="last")

        labels_df["__order__"] = range(len(labels_df))
        merged = labels_df.merge(df, on="Fecha", how="left").sort_values("__order__").drop(columns="__order__")
        merged.to_csv(out_csv, index=False)

        self.logger.info("[ENS/%s] Labels created and aligned by date -> %s | %s", subset, lab_npz, out_csv)

        return {
            "subset": subset,
            "n": int(best_model_indices.size),
            "npz": lab_npz,
            "json": lab_json,
            "csv": out_csv,
            "ranking": ranking,
        }

    def compute_metrics_for_all_models(self,
                                       metrics: Optional[List[str]] = None,
                                       force: bool = False,
                                       save_summary: bool = True) -> pd.DataFrame:
        """Evaluate every fine-tuned model's test predictions with the metric registry.

        Args:
            metrics (Optional[List[str]]): Keys of ``METRICS_REGISTRY``; defaults
                to overall, top-10%, alarm and timing metrics.
            force (bool): Recompute even if a metrics JSON exists.
            save_summary (bool): Write ``metrics_summary.csv``.

        Returns:
            pd.DataFrame: One summary row per model.

        Raises:
            RuntimeError: If the training threshold is not precomputed.
        """
        if metrics is None:
            metrics = ["overall_hydroeval", "top10_hits_misses", "alarm_metrics", "timing_error"]

        preds = self.ensure_predictions(force=False, save_csv=True)
        raw_thr = self.p.get("eval_threshold", "p90")
        threshold_key = raw_thr[0] if isinstance(raw_thr, list) else raw_thr

        rows = []
        for pr in preds:
            base = pr["base"]
            npz  = pr["npz"]
            metf = os.path.join(self.paths.predictions_cache, f"metrics_{base}.json")

            if os.path.exists(metf) and not force:
                with open(metf, "r", encoding="utf-8") as f:
                    met = json.load(f)
            else:
                data = np.load(npz, allow_pickle=True)
                y_true = data["y_true"]
                y_pred = data["y_pred"]

                # Anti-leakage: falling back to np.quantile(y_true, ...) on TEST
                # is forbidden; a missing training threshold must fail loudly.
                if not self._norm_params or "Qe" not in self._norm_params or threshold_key not in self._norm_params["Qe"]:
                    raise RuntimeError(
                        f"[ANTI-LEAKAGE] Data leakage prevented: threshold {threshold_key} is not "
                        f"precomputed in norm_params. Computing quantiles on y_true during evaluation is forbidden."
                    )
                thr_val = float(self._norm_params["Qe"][threshold_key])

                met = {}
                for key in metrics:
                    fn = METRICS_REGISTRY.get(key)
                    if fn is None:
                        self.logger.warning("[METRICS] Metric '%s' is not registered. Skipping.", key)
                        continue
                    try:
                        kwargs = {"y_true": y_true, "y_pred": y_pred, "dates": data.get("dates", None)}
                        sig = inspect.signature(fn)
                        if "threshold" in sig.parameters:
                            kwargs["threshold"] = thr_val
                        if "train_p90_threshold" in sig.parameters:
                            kwargs["train_p90_threshold"] = thr_val

                        part = fn(**kwargs)

                        for k, v in part.items():
                            if isinstance(v, dict) and isinstance(met.get(k), dict):
                                met[k].update(v)
                            else:
                                met[k] = v
                    except Exception as e:
                        self.logger.warning("[METRICS] '%s' failed for %s: %s", key, base, e)
                with open(metf, "w", encoding="utf-8") as f:
                    json.dump(met, f, ensure_ascii=False, indent=2)

            row = {
                "model": base,
                "overall_NSE": met.get("overall", {}).get("NSE"),
                "overall_KGE": met.get("overall", {}).get("KGE"),
                "top10_n": met.get("top10", {}).get("n"),
                "top10_hits": met.get("top10", {}).get("hits"),
                "top10_misses": met.get("top10", {}).get("misses"),
                "top10_hit_ratio": met.get("top10", {}).get("hit_ratio"),
                "top10_miss_ratio": met.get("top10", {}).get("miss_ratio"),
                "top10_NSE": met.get("top10", {}).get("NSE"),
                "top10_KGE": met.get("top10", {}).get("KGE"),
            }
            rows.append(row)

        summary = pd.DataFrame(rows).sort_values("model")

        if save_summary:
            out_csv = os.path.join(self.paths.predictions_cache, "metrics_summary.csv")
            summary.to_csv(out_csv, index=False)
            self.logger.info("[METRICS] Summary saved to %s", out_csv)

        return summary

    def _classifiers_dir(self) -> str:
        """Directory of the trained meta-learners, created if needed.

        Returns:
            str: The directory path.
        """
        d = os.path.join(self.paths.ensemble, "classifiers")
        os.makedirs(d, exist_ok=True)
        return d

    def _classifier_paths(self, algo: str) -> Dict[str, str]:
        """File paths of a meta-learner and its outputs.

        Args:
            algo (str): Meta-learner name (e.g. ``rf_regressor``).

        Returns:
            Dict[str, str]: ``keras``, ``pkl``, ``meta``, ``pred_npz``,
            ``pred_csv`` and ``metrics_json`` paths.
        """
        base = os.path.join(self._classifiers_dir(), f"{algo}")
        return {
            "keras": base + ".keras",
            "pkl":   base + ".pkl",
            "meta":  base + ".json",
            "pred_npz": os.path.join(self.paths.predictions_cache, f"classifier_{algo}.npz"),
            "pred_csv": os.path.join(self.paths.predictions_cache, f"classifier_{algo}.csv"),
            "metrics_json": os.path.join(self.paths.predictions_cache, f"classifier_{algo}_metrics.json"),
        }

    def _save_classifier(self, clf: Any, algo: str, model_bases: List[str], input_context_shape: Tuple[int, ...], input_future_shape: Tuple[int, ...]) -> Dict[str, str]:
        """Persist a meta-learner (Keras or pickle) and its metadata.

        Args:
            clf (Any): Trained meta-learner.
            algo (str): Meta-learner name.
            model_bases (List[str]): Candidate model order used in training.
            input_context_shape (Tuple[int, ...]): Encoder window shape.
            input_future_shape (Tuple[int, ...]): Decoder window shape.

        Returns:
            Dict[str, str]: The classifier paths.
        """
        paths = self._classifier_paths(algo)
        try:
            if isinstance(clf, tf.keras.Model):
                clf.save(paths["keras"])
            elif hasattr(clf, "save") and callable(getattr(clf, "save")):
                clf.save(paths["pkl"])
            else:
                with open(paths["pkl"], "wb") as f:
                    pickle.dump(clf, f)
        except Exception:
            with open(paths["pkl"], "wb") as f:
                pickle.dump(clf, f)
        meta = {
            "algo": algo,
            "num_models": int(len(model_bases)),
            "model_bases": list(model_bases),
            "input_context_shape": list(input_context_shape),
            "input_future_shape": list(input_future_shape),
            "offsets": int(self.p["offsets"]),
        }
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return paths

    def _load_classifier(self, algo: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Load a saved meta-learner and its metadata.

        Args:
            algo (str): Meta-learner name.

        Returns:
            Tuple[Optional[Any], Optional[Dict[str, Any]]]: ``(classifier, meta)``,
            or ``(None, None)`` if not found.
        """
        paths = self._classifier_paths(algo)
        clf = None
        if os.path.exists(paths["keras"]):
            clf = tf.keras.models.load_model(paths["keras"])
        elif os.path.exists(paths["pkl"]):
            with open(paths["pkl"], "rb") as f:
                clf = pickle.load(f)
        if clf is None:
            return None, None
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        return clf, meta

    def _classifier_registry(self) -> Dict[str, Callable[..., Any]]:
        """Available meta-learner builders.

        Optional legacy builders from another experiment folder are registered
        only when importable; they are not part of this repository.

        Returns:
            Dict[str, Callable[..., Any]]: Name -> builder.
        """
        reg = {
            "random_forest": build_random_forest_classifier,
            "xgboost":       build_xgboost_classifier,
            "rf_regressor":  build_random_forest_regressor,
        }
        try:
            from flowpredictor.resultados.prueba60.src.models.models import build_stacked_lstm_classifier
            reg["stacked_lstm"] = build_stacked_lstm_classifier
        except Exception:
            pass
        try:
            from flowpredictor.resultados.prueba60.src.models.models import build_tcn_classifier
            reg["tcn"] = build_tcn_classifier
        except Exception:
            pass
        return reg

    def train_classifier(self, force: bool = False) -> Dict[str, Any]:
        """Train the M5 meta-learner on genuine out-of-fold predictions.

        OOF predictions cover a subset of the training dates (the burn-in is
        excluded under ``"expanding"``), so encoder/decoder windows are aligned
        to that subset by date lookup, never by position. The regression
        target is the label already computed by
        :meth:`ensure_ensemble_labels`; since every row is out-of-sample with
        respect to the base model that produced it, the classifier is fitted
        on all of them.

        Args:
            force (bool): Retrain even if a saved classifier exists.

        Returns:
            Dict[str, Any]: ``algo``, ``paths`` and ``meta``.

        Raises:
            RuntimeError: If dates or row counts are misaligned.
        """
        algo = str(self.p.get("algorithms", "rf_regressor"))
        reg  = self._classifier_registry()
        seed = int(self.p.get("seed", 42))

        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

        clf_loaded, meta = self._load_classifier(algo)
        if clf_loaded is not None and not force:
            return {"algo": algo, "paths": self._classifier_paths(algo), "meta": meta}

        (_, _, _, _), (x_te_enc, x_te_dec, _, _) = self.build_windows_normal()
        x_tr_enc, x_tr_dec, y_tr, tr_dates = self.train_inputs

        ens_train = self.ensure_ensemble_labels(subset="train", force=False)
        z = np.load(ens_train["npz"], allow_pickle=True)
        model_bases = [ _canonical_base(b) for b in z["models"].tolist() ]

        pred_list = self.ensure_oof_train_predictions(force=False)
        pred_map = { _canonical_base(pr["base"]): pr for pr in pred_list }
        ordered = [pred_map[b] for b in model_bases]

        y_true_train = None
        dates_train_preds = None
        preds_stack = []
        for pr in ordered:
            d = np.load(pr["npz"], allow_pickle=True)
            if y_true_train is None:
                y_true_train = d["y_true"].astype(float)
                dates_train_preds = pd.to_datetime(d["dates"])
            preds_stack.append(d["y_pred"].astype(float))

        predictions_all = np.vstack(preds_stack)
        base_preds_train = predictions_all.T

        num_models = len(model_bases)
        N = base_preds_train.shape[0]

        # Anti-shuffle: OOF predictions cover a subset of the train_inputs dates,
        # so x_tr_enc/x_tr_dec are aligned to that subset by date (lookup)
        # rather than by position after a parallel argsort.
        dates_windows = pd.to_datetime(pd.Index(list(tr_dates)))
        if dates_windows.shape[0] != x_tr_enc.shape[0]:
            raise RuntimeError(
                f"[ANTI-SHUFFLE] tr_dates ({dates_windows.shape[0]}) does not match the length "
                f"of x_tr_enc ({x_tr_enc.shape[0]})."
            )
        if dates_train_preds is None or dates_train_preds.shape[0] != N:
            raise RuntimeError(
                "[ANTI-SHUFFLE] Could not read the OOF dates of the base predictions "
                "('dates' field of the .npz) or their length does not match base_preds_train."
            )
        if not set(dates_train_preds).issubset(set(dates_windows)):
            raise RuntimeError(
                "[ANTI-SHUFFLE] The OOF dates of the base predictions are not a subset "
                "of the train_inputs dates (x_tr_enc/x_tr_dec) -- temporal alignment "
                "cannot be guaranteed."
            )

        order_preds = np.argsort(dates_train_preds.values, kind="stable")
        base_preds_train = base_preds_train[order_preds]
        y_true_train = y_true_train[order_preds]
        dates_train_preds = dates_train_preds[order_preds]

        date_to_window_idx = {d: i for i, d in enumerate(dates_windows)}
        window_idx_for_oof = np.array([date_to_window_idx[d] for d in dates_train_preds])
        x_tr_enc = x_tr_enc[window_idx_for_oof]
        x_tr_dec = x_tr_dec[window_idx_for_oof]

        # Anti-leakage: the OFFICIAL label is the one already computed by
        # ensure_ensemble_labels() (formal evaluation of the configured
        # asymmetric loss), not a heuristic recomputed here.
        target_indices = z["indices"].astype(float)
        if target_indices.shape[0] != N:
            raise RuntimeError(
                f"[CLS] Misalignment between target_indices ({target_indices.shape[0]}) and "
                f"base_preds_train ({N}): ensure_ensemble_labels()/ensure_oof_train_predictions() "
                f"must produce the same number of OOF rows in train."
            )
        # target_indices was computed on the same raw order returned by
        # ensure_oof_train_predictions (that of dates_train_preds before
        # sorting), so it is reordered with the same order_preds.
        target_indices = target_indices[order_preds]

        # Every row of base_preds_train/x_tr_enc/x_tr_dec is genuinely
        # out-of-sample with respect to the base model that produced it, so no
        # chronological hold-out is reserved: the classifier uses all of them.
        self.logger.info(
            "[CLS] Fitting the classifier on %d genuine OOF samples "
            "(Nested Blocked Out-Of-Fold, cv_strategy=%s).",
            N, self.p.get("cv_strategy", "expanding"),
        )

        input_context_shape = x_tr_enc.shape[1:]
        input_future_shape  = x_tr_dec.shape[1:]
        build_fn = reg[algo]
        clf = build_fn(input_context_shape, input_future_shape, num_models, self.p)

        self.logger.info("[CLS] Training the meta-learner (ordinal regression)…")
        clf.fit([x_tr_enc, x_tr_dec, base_preds_train], target_indices)

        paths = self._save_classifier(clf, algo, model_bases, input_context_shape, input_future_shape)
        return {"algo": algo, "paths": paths, "meta": {"model_bases": model_bases, "num_models": num_models}}

    def predict_classifier(self, force: bool = False) -> Dict[str, Any]:
        """Select, per test time step, the branch chosen by the meta-learner.

        The meta-learner outputs a continuous branch index, rounded and
        clipped to the valid range; the served prediction is that branch's
        prediction. Branches are stacked in the order used for training
        (``meta["model_bases"]``).

        Args:
            force (bool): Unused; kept for API symmetry.

        Returns:
            Dict[str, Any]: ``paths`` and number of predictions ``N``.

        Raises:
            RuntimeError: If the training threshold is not precomputed.
        """
        algo = str(self.p.get("algorithms", "rf_regressor"))
        clf, meta = self._load_classifier(algo)

        if clf is None:
            self.train_classifier(force=False)
            clf, meta = self._load_classifier(algo)

        paths = self._classifier_paths(algo)
        have_normal = (hasattr(self, "test_inputs") and getattr(self, "_windows_tag", None) == "normal")
        if not have_normal:
            (_, _, _, _), (x_te_enc, x_te_dec, y_te, te_dates) = self.build_windows_normal()
        else:
            x_te_enc, x_te_dec, y_te, te_dates = self.test_inputs

        bases_want = [_canonical_base(b) for b in meta["model_bases"]]
        pred_list = self.ensure_predictions(force=False, save_csv=True, subset="test")
        pred_map = { _canonical_base(pr["base"]): pr for pr in pred_list }
        ordered = [pred_map[b] for b in bases_want]

        y_true, dates = None, None
        preds_stack = []
        for pr in ordered:
            d = np.load(pr["npz"], allow_pickle=True)
            if y_true is None:
                y_true = d["y_true"].astype(float)
                dates = pd.to_datetime(d["dates"])
            preds_stack.append(d["y_pred"].astype(float))

        predictions_all = np.vstack(preds_stack)
        base_preds_test = predictions_all.T

        N = base_preds_test.shape[0]
        num_models = predictions_all.shape[0]

        if x_te_enc.shape[0] != N:
            x_te_enc = x_te_enc[-N:]
            x_te_dec = x_te_dec[-N:]
            base_preds_test = base_preds_test[-N:]

        predicted_continuous_idx = clf.predict([x_te_enc, x_te_dec, base_preds_test])
        cls_idx = np.round(predicted_continuous_idx).astype(int)
        cls_idx = np.clip(cls_idx, 0, num_models - 1)
        y_cls = predictions_all[cls_idx, np.arange(N)]

        np.savez_compressed(
            paths["pred_npz"],
            dates=np.array(dates.astype('datetime64[ns]')).astype('datetime64[ns]'),
            y_true=y_true,
            y_pred=y_cls,
            cls_idx=cls_idx,
            predicted_scores=predicted_continuous_idx,
            model_bases=np.array(bases_want, dtype=object),
        )

        # Anti-leakage: the classifier uses the stored training p90 and never
        # recomputes it on test.
        raw_thr = self.p.get("eval_threshold", "p90")
        threshold_key = raw_thr[0] if isinstance(raw_thr, list) else raw_thr
        if not self._norm_params or "Qe" not in self._norm_params or threshold_key not in self._norm_params["Qe"]:
            raise RuntimeError("[ANTI-LEAKAGE] Computing dynamic quantiles on Test is forbidden.")
        thr = float(self._norm_params["Qe"][threshold_key])

        mask_top = (y_true >= thr).astype(int)
        hit = (y_cls >= y_true).astype(int)

        df = pd.DataFrame({
            "Date": dates,
            "Observed": y_true,
            "Classifier": y_cls,
            "TopMask": mask_top,
            "Classifier_Hit": hit,
        })
        df.to_csv(paths["pred_csv"], index=False)
        self.logger.info("[CLS] Meta-learner selected the winning models from their own predictions.")
        return {"paths": paths, "N": int(N)}

    def evaluate_classifier(self, metrics: Optional[List[str]] = None, force: bool = False) -> Dict[str, Any]:
        """Evaluate the meta-learner's test predictions with the metric registry.

        Args:
            metrics (Optional[List[str]]): Keys of ``METRICS_REGISTRY``.
            force (bool): Recompute even if the metrics JSON exists.

        Returns:
            Dict[str, Any]: Merged metric results.

        Raises:
            RuntimeError: If the training threshold is not precomputed.
        """
        if metrics is None:
            metrics = ["overall_hydroeval", "top10_hits_misses", "alarm_metrics", "timing_error"]

        algo = str(self.p.get("algorithms", "rf_regressor"))
        paths = self._classifier_paths(algo)

        if not os.path.exists(paths["pred_npz"]):
            self.predict_classifier(force=False)

        if os.path.exists(paths["metrics_json"]) and not force:
            with open(paths["metrics_json"], "r", encoding="utf-8") as f:
                return json.load(f)

        data = np.load(paths["pred_npz"], allow_pickle=True)
        y_true = data["y_true"].astype(float)
        y_pred = data["y_pred"].astype(float)

        # Anti-leakage: falling back to np.quantile(y_true, ...) on TEST is
        # forbidden; a missing training threshold must fail loudly.
        raw_thr = self.p.get("eval_threshold", "p90")
        threshold_key = raw_thr[0] if isinstance(raw_thr, list) else raw_thr
        if not self._norm_params or "Qe" not in self._norm_params or threshold_key not in self._norm_params["Qe"]:
            raise RuntimeError(
                f"[ANTI-LEAKAGE] Data leakage prevented: threshold {threshold_key} is not "
                f"precomputed in norm_params. Computing quantiles on y_true during evaluation is forbidden."
            )
        thr_val = float(self._norm_params["Qe"][threshold_key])

        met = {}
        for key in metrics:
            fn = METRICS_REGISTRY.get(key)
            if fn is None:
                self.logger.warning("[METRICS/CLS] Metric '%s' is not registered. Skipping.", key)
                continue
            try:
                kwargs = {"y_true": y_true, "y_pred": y_pred, "dates": data.get("dates", None)}
                sig = inspect.signature(fn)
                if "threshold" in sig.parameters:
                    kwargs["threshold"] = thr_val
                if "train_p90_threshold" in sig.parameters:
                    kwargs["train_p90_threshold"] = thr_val

                part = fn(**kwargs)

                for k, v in part.items():
                    if isinstance(v, dict) and isinstance(met.get(k), dict):
                        met[k].update(v)
                    else:
                        met[k] = v
            except Exception as e:
                self.logger.warning("[METRICS/CLS] '%s' failed: %s", key, e)

        with open(paths["metrics_json"], "w", encoding="utf-8") as f:
            json.dump(met, f, ensure_ascii=False, indent=2)

        self.logger.info("[METRICS/CLS] Metrics saved to %s", paths["metrics_json"])
        return met
