"""Small helpers: scalar extraction, stable configuration hashing and log sanitization."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import numpy as np


def _scalar(value: Any, default: Any) -> Any:
    """Return ``default`` for ``None``, the first element of a list/tuple, or the value itself."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0]
    return value

def sha1_of_dict(d: Dict) -> str:
    """Stable SHA-1 of a (possibly nested) dict, independent of key order.

    Callables are represented by their name and arrays by shape and dtype.

    Args:
        d (Dict): Payload.

    Returns:
        str: Hex digest.
    """
    def _canon(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _canon(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [_canon(v) for v in x]
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if callable(x):
            return getattr(x, "__name__", str(x))
        if isinstance(x, np.ndarray):
            return {"__ndarray__": True, "shape": x.shape, "dtype": str(x.dtype)}
        return str(x)
    payload = _canon(d)
    b = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(b).hexdigest()

def sanitize_for_log(obj: Any) -> Any:
    """Convert non-serializable objects (callables, ndarrays, numpy scalars) into loggable values.

    Args:
        obj (Any): Object, possibly nested.

    Returns:
        Any: JSON-friendly equivalent.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_log(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_log(v) for v in obj]
    if callable(obj):
        return getattr(obj, "__name__", str(obj))
    if isinstance(obj, np.ndarray):
        return f"ndarray{list(obj.shape)}"
    # numpy scalars
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj

