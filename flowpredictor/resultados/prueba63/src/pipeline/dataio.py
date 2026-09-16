"""Data-bundle I/O: locate, generate, copy and load normalized train/test bundles."""
from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Dict, Tuple

import pandas as pd

from ..data.dataset_generator import run as data_generator


def resolve_bundle_base(CONFIG: Any, split: str, test_year: int) -> str:
    """Directory of the bundle for a split label and test year.

    Args:
        CONFIG (Any): Pipeline configuration (``PipelineConfig``).
        split (str): Split label (e.g. ``JunioExpanding``).
        test_year (int): Test year.

    Returns:
        str: ``<data_bundle_root>/<split>/test<test_year>``.
    """
    return os.path.join(CONFIG.paths["data_bundle_root"], split, f"test{test_year}")


def bundle_manifest_path(bundle_base: str) -> str:
    """Path of a bundle's optional manifest.

    Args:
        bundle_base (str): Bundle directory.

    Returns:
        str: ``<bundle_base>/manifest.json``.
    """
    return os.path.join(bundle_base, "manifest.json")


def bundle_has_normalized(bundle_base: str) -> bool:
    """Whether a bundle has its normalized partitions and normalization parameters.

    Args:
        bundle_base (str): Bundle directory.

    Returns:
        bool: ``True`` if train, test and normalization parameters exist.
    """
    # Normalized train/test + normalization_params are required; the manifest may not exist.
    has_norm = (
        os.path.exists(os.path.join(bundle_base, "normalized", "train.csv")) and
        os.path.exists(os.path.join(bundle_base, "normalized", "test.csv")) and
        os.path.exists(os.path.join(bundle_base, "normalization_params.json"))
    )
    return has_norm


def copy_bundle_subset_to_iteration(bundle_base: str, iter_data_dir: str) -> None:
    """Copy a bundle's normalized partitions (and manifest, if any) into an iteration.

    Args:
        bundle_base (str): Bundle directory.
        iter_data_dir (str): Iteration data directory (created if needed).
    """
    os.makedirs(iter_data_dir, exist_ok=True)
    # Source paths
    src_train = os.path.join(bundle_base, "normalized", "train.csv")
    src_test  = os.path.join(bundle_base, "normalized", "test.csv")
    src_norm  = os.path.join(bundle_base, "normalization_params.json")
    src_manifest = bundle_manifest_path(bundle_base)

    # Destination paths
    dst_train = os.path.join(iter_data_dir, "train.csv")
    dst_test  = os.path.join(iter_data_dir, "test.csv")
    dst_norm  = os.path.join(iter_data_dir, "normalization_params.json")
    dst_manifest = os.path.join(iter_data_dir, "manifest.json")

    # Required copies
    shutil.copy2(src_train, dst_train)
    shutil.copy2(src_test,  dst_test)
    shutil.copy2(src_norm,  dst_norm)

    # Optional manifest copy (only if the bundle has one)
    if os.path.exists(src_manifest):
        shutil.copy2(src_manifest, dst_manifest)


def ensure_iteration_dataset(CONFIG: Any, split: str, test_year: int, iter_data_dir: str, logger: logging.Logger) -> None:
    """Make sure the bundle exists (generating it if needed) and copy it into the iteration.

    Args:
        CONFIG (Any): Pipeline configuration (``PipelineConfig``).
        split (str): Split label.
        test_year (int): Test year.
        iter_data_dir (str): Iteration data directory.
        logger (logging.Logger): Logger.

    Raises:
        FileNotFoundError: If the bundle is still missing after generation.
    """
    bundle_base = resolve_bundle_base(CONFIG, split, test_year)
    if not bundle_has_normalized(bundle_base):
        logger.info(f"[DATA] No normalized bundle for split={split}, test_year={test_year}. Generating it with data_generator()…")
        data_generator(CONFIG)
    else:
        logger.info(f"[DATA] Normalized bundle found for split={split}, test_year={test_year}. Loading from disk")
    # Re-check after generation
    if not bundle_has_normalized(bundle_base):
        raise FileNotFoundError(f"Normalized files not found in {bundle_base} after running data_generator().")
    copy_bundle_subset_to_iteration(bundle_base, iter_data_dir)
    logger.info(f"[DATA] Dataset copied to {iter_data_dir}")


def _read_json(path: str) -> Dict:
    """Read a UTF-8 JSON file.

    Args:
        path (str): File path.

    Returns:
        Dict: Parsed content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_iteration_bundle(iter_data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """Load the normalized partitions, normalization parameters and optional manifest.

    Args:
        iter_data_dir (str): Iteration data directory.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        ``(train_df, test_df, normalization_params, manifest)``; the manifest
        is an empty dict when absent.

    Raises:
        FileNotFoundError: If a required file is missing.
    """
    train_path = os.path.join(iter_data_dir, "train.csv")
    test_path  = os.path.join(iter_data_dir, "test.csv")
    norm_path  = os.path.join(iter_data_dir, "normalization_params.json")
    manifest_path = os.path.join(iter_data_dir, "manifest.json")

    for p in [train_path, test_path, norm_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    normalization = _read_json(norm_path)
    manifest = _read_json(manifest_path) if os.path.exists(manifest_path) else {}

    return train_df, test_df, normalization, manifest
