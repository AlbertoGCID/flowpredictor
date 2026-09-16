"""Directory layout of an iteration."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class IterationPaths:
    """Paths of the artifacts of one iteration (``tests<N>/<hash_id>/...``)."""
    root: str
    state: str
    data: str
    models: str
    pretrain: str
    ensemble: str
    predictions_cache: str

def ensure_dirs(paths: IterationPaths) -> None:
    """Create every directory of an iteration layout."""
    for p in [paths.root, paths.data, paths.models, paths.pretrain, paths.ensemble, paths.predictions_cache]:
        os.makedirs(p, exist_ok=True)
