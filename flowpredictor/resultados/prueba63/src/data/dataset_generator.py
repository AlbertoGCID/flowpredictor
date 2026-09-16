"""Dataset generation: gap filling, leak-free temporal splits and normalized bundles.

Each bundle (``<root_out>/<split>/test<YEAR>/``) contains the raw and normalized
train/test partitions, the normalization parameters computed on TRAIN only
(excluding imputed rows, including the p90/p95 extreme-event thresholds) and a
manifest with SHA-256 hashes.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import json
import tensorflow as tf
import numpy as np
import itertools
import glob
import shutil
from statistics import mean

# Directory that contains the project root folder (five levels above this file).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5)))
import pandas as pd
from ..config import CONFIG
import logging

import hashlib
from dataclasses import dataclass, asdict
from typing import Any, List, Tuple, Dict, Optional, Union

# ----------------- BASIC CONFIGURATION -----------------
# Values from experiments_config.toml [dataset]/[holdout]. The dataset paths are
# relative to the parent directory of the project root folder, which must be
# the cwd when this module runs.
ROOT_RAW = str(CONFIG.dataset["root_raw"])
ROOT_OUT = str(CONFIG.dataset["root_out"])   # root where <split>/testYYYY/ bundles are created
RANDOM_SEED = int(CONFIG.dataset["random_seed"])
NOISE_FRAC = float(CONFIG.dataset["noise_frac"])  # ±10% multiplicative noise for imputed rows
TARGET_COL = str(CONFIG.dataset["target_col"])

# Force specific test years here (None => detect complete years)
TEST_YEARS: Optional[List[int]] = None  # e.g. [2014, 2015, ..., 2023]

# Holdout cutoff (D_holdout = 2024-07-01 onwards): run() (LOYO bundles
# Diciembre/Junio) must never produce a fold whose training set contains holdout
# data. Unlike the expanding CV, LOYO uses years BEFORE and AFTER the test year
# (see load_and_split_data(expanding_window=False)), so without this cutoff any
# test_year <= 2023 would leak 2024/2025 into its own training set.
HOLDOUT_CUTOFF_DATE = pd.Timestamp(CONFIG.holdout["cutoff_date"])

# Rain/forecast column prefixes, filled with 0 for missing days of the first subset
RAIN_PRED_PREFIXES = tuple(CONFIG.dataset["rain_pred_prefixes"])

# -------------------------------------------------


def setup_logger() -> logging.Logger:
    """Configure basic logging and return the module logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("pipeline52")


logger = setup_logger()


# ----------------- UTILITIES -----------------
def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def sha256_file(path: str) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _normalize_split_value(split: Union[str, List[str]]) -> str:
    """Normalize a split given as ``"Diciembre"`` or ``["Diciembre"]`` to a string."""
    if isinstance(split, list):
        return split[0]
    return split


def _year_range_for_split(test_year: int, split: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Evaluation date range of a test year.

    Args:
        test_year (int): Test year label.
        split (str): ``"Diciembre"`` (calendar year) or ``"Junio"`` (hydrological
            year July ``test_year`` -> June ``test_year + 1``).

    Returns:
        Tuple[pd.Timestamp, pd.Timestamp]: Inclusive start and end dates.

    Raises:
        ValueError: If the split is unknown.
    """
    if split == "Diciembre":
        return (pd.Timestamp(f"{test_year}-01-01"), pd.Timestamp(f"{test_year}-12-31"))
    elif split == "Junio":
        return (pd.Timestamp(f"{test_year}-07-01"), pd.Timestamp(f"{test_year+1}-06-30"))
    else:
        raise ValueError(f"invalid split: {split}")


# ----------------- GAP FILLING -----------------
def rellenar_huecos_intermedios_determinista(df: pd.DataFrame,
                                              seed: int,
                                              noise_frac: float,
                                              target_col: str) -> pd.DataFrame:
    """Fill missing dates by linear interpolation plus deterministic noise.

    Missing rows (by ``Fecha``) are interpolated linearly, and deterministic
    multiplicative noise ``±noise_frac`` is added to EVERY numeric column
    (including ``target_col``) ONLY in the imputed rows, which are flagged in
    ``is_imputed``.

    Args:
        df (pd.DataFrame): Daily data with ``Fecha``.
        seed (int): Noise seed.
        noise_frac (float): Noise amplitude.
        target_col (str): Target column name.

    Returns:
        pd.DataFrame: Complete daily series with ``is_imputed``.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["Fecha"] = pd.to_datetime(out["Fecha"])
    out = out.sort_values("Fecha").reset_index(drop=True)

    full_range = pd.date_range(out["Fecha"].min(), out["Fecha"].max())
    missing = full_range.difference(out["Fecha"])

    # No missing dates -> only flag is_imputed and return
    if len(missing) == 0:
        out["is_imputed"] = False
        return out

    miss_df = pd.DataFrame({"Fecha": missing})
    miss_df["is_imputed"] = True
    out["is_imputed"] = False

    all_df = pd.concat([out, miss_df], ignore_index=True).sort_values("Fecha").reset_index(drop=True)
    mask = all_df["Fecha"].isin(missing)

    # Interpolate and add noise to imputed rows (Qe included)
    cols = [c for c in out.columns if c not in ("Fecha", "is_imputed")]
    for col in cols:
        all_df[col] = all_df[col].interpolate(method="linear")
        ruido = rng.uniform(-noise_frac, noise_frac, size=mask.sum())  # imputed rows only
        # multiplicative: interpolated_value * (1 ± noise)
        all_df.loc[mask, col] = all_df.loc[mask, col] * (1.0 + ruido)

    return all_df


def rellenar_faltantes_primer_subconjunto(df: pd.DataFrame,
                                          start_date: pd.Timestamp,
                                          end_date: pd.Timestamp,
                                          rain_pred_prefixes: Tuple[str, ...],
                                          target_col: str) -> pd.DataFrame:
    """Fill missing days of the first subset conservatively.

    - ``target_col`` := minimum Qe of the DataFrame (conservative)
    - rain/forecast columns := 0

    The filled rows are flagged with ``is_imputed=True``.

    Args:
        df (pd.DataFrame): Daily data.
        start_date (pd.Timestamp): Subset start.
        end_date (pd.Timestamp): Subset end.
        rain_pred_prefixes (Tuple[str, ...]): Rain/forecast column prefixes.
        target_col (str): Target column name.

    Returns:
        pd.DataFrame: Data with the missing days added.
    """
    out = df.copy()
    sub = out[(out["Fecha"] >= start_date) & (out["Fecha"] <= end_date)]
    full = pd.date_range(start_date, end_date)
    missing = full.difference(sub["Fecha"])
    if len(missing) == 0:
        return out

    logger.info(f"Filling {len(missing)} missing days in the first subset {start_date.date()}–{end_date.date()}")
    min_qe = out[target_col].min()
    add = pd.DataFrame({"Fecha": missing})
    add[target_col] = min_qe

    # 0 for rain/forecast columns
    for col in out.columns:
        if any(col.startswith(p) for p in rain_pred_prefixes):
            add[col] = 0

    add["is_imputed"] = True
    if "is_imputed" not in out.columns:
        out["is_imputed"] = False

    out = pd.concat([out, add], ignore_index=True).sort_values("Fecha").reset_index(drop=True)
    return out


# ----------------- SPLIT -----------------
def load_and_split_data(df: pd.DataFrame,
                        test_year: int,
                        split: str,
                        context_days: list = [30],
                        expanding_window: bool = False,
                        eval_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset with a ``context_days`` gap to prevent data leakage.

    - ``expanding_window=False``: train = everything outside the test window,
      years both BEFORE and AFTER the test year (leave-one-year-out with a
      context gap).
    - ``expanding_window=True``: train = only the years BEFORE the test year
      (Train=[dataset start, T-1], Test=T), never future information. Used for
      the 2017-2023 expanding-window validation and the final holdout.

    ``eval_range=(eval_start, eval_end)`` replaces
    ``_year_range_for_split(test_year, split)`` with an explicit date range,
    needed for D_holdout (continuous block from 2024-07-01 to the end of the
    real data). It requires ``expanding_window=True`` (the holdout must never
    leak into train). ``test_year`` is then only the bundle directory label.

    Args:
        df (pd.DataFrame): Daily data with ``Fecha``.
        test_year (int): Test year label.
        split (str): ``"Diciembre"`` or ``"Junio"``.
        context_days (list): Encoder context length (first element used).
        expanding_window (bool): Use only past years for training.
        eval_range (Optional[Tuple[pd.Timestamp, pd.Timestamp]]): Explicit evaluation range.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: ``(train_data, test_data)``; the test
        set includes the ``context_days`` preceding the evaluation start.

    Raises:
        ValueError: On an invalid argument combination or an empty partition.
        AssertionError: If a leakage check fails.
    """
    if eval_range is not None and not expanding_window:
        raise ValueError("eval_range requires expanding_window=True (the holdout must never leak into train).")

    split = _normalize_split_value(split)
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    logger.info(f"[DEBUG] Window size {context_days}")

    if isinstance(context_days, (list, tuple)):
        context_days = int(context_days[0])
    else:
        context_days = int(context_days)

    # Exact range to EVALUATE (e.g. 2013-07-01 to 2014-06-30), or the explicit
    # eval_range if given (D_holdout).
    eval_start, eval_end = eval_range if eval_range is not None else _year_range_for_split(test_year, split)

    # 1. TEST SET:
    # The preceding 'context_days' are physically included so that the window
    # generator can produce the prediction for day 1 (eval_start).
    test_df_start = eval_start - pd.Timedelta(days=context_days)
    test_data = df[(df["Fecha"] >= test_df_start) & (df["Fecha"] <= eval_end)]

    # 2. TRAINING SET (EXCLUSION ZONE):
    # Train ends one day before the test window starts (e.g. 2013-05-31).
    train_end_before = test_df_start - pd.Timedelta(days=1)

    if expanding_window:
        # Expanding window: ONLY the past, never data after the test year.
        train_data = df[df["Fecha"] <= train_end_before]
    else:
        # Train resumes once a day's history no longer overlaps the test year (e.g. 2014-07-30).
        train_start_after = eval_end + pd.Timedelta(days=context_days)
        # Train is the union of what lies BEFORE and AFTER the gap
        train_data = df[(df["Fecha"] <= train_end_before) | (df["Fecha"] >= train_start_after)]

    if test_data.empty:
        raise ValueError(f"empty test_data for {split} {test_year}")
    if train_data.empty:
        raise ValueError(f"empty train_data for {split} {test_year}")

    # Explicit leakage checks
    max_train_before = train_data[train_data["Fecha"] <= train_end_before]["Fecha"].max()

    if pd.notna(max_train_before):
        assert max_train_before < test_df_start, "Data Leakage: train overlaps the historical window of the test set."

    if expanding_window:
        assert train_data["Fecha"].max() < test_df_start, \
            "Data Leakage: expanding_window=True but train contains dates after the test set."
    else:
        train_start_after = eval_end + pd.Timedelta(days=context_days)
        min_train_after = train_data[train_data["Fecha"] >= train_start_after]["Fecha"].min()
        if pd.notna(min_train_after):
            assert min_train_after > eval_end, "Data Leakage: train reads data inside the test evaluation range."

    return train_data, test_data


# ----------------- NORMALIZATION AND THRESHOLDS -----------------
def calculate_normalization_params_excl_imputed(train_df: pd.DataFrame,
                                                target_col: str) -> Dict[str, Dict[str, float]]:
    """Per-column min/max/range and PERCENTILES (p90, p95), excluding imputed rows.

    Storing the p90/p95 computed on TRAIN removes data leakage when extremes
    are evaluated on TEST.

    Args:
        train_df (pd.DataFrame): Training partition.
        target_col (str): Target column name (kept for API compatibility).

    Returns:
        Dict[str, Dict[str, float]]: ``{column: {min, max, range, p90, p95}}``.
    """
    df = train_df.copy()
    if "is_imputed" in df.columns:
        df = df[~df["is_imputed"]]

    df = df.drop(columns=["Fecha", "is_imputed"], errors="ignore")

    params = {}
    for col in df.columns:
        col_min = float(df[col].min())
        col_max = float(df[col].max())

        # Fixed percentiles for rigorous extreme-event validation
        col_p90 = float(df[col].quantile(0.90))
        col_p95 = float(df[col].quantile(0.95))

        params[col] = {
            "min": col_min, 
            "max": col_max, 
            "range": (col_max - col_min),
            "p90": col_p90,
            "p95": col_p95
        }
    return params


def normalize_with_params(df: pd.DataFrame, params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Min-max normalize with precomputed parameters (``Fecha``/``is_imputed`` kept).

    Args:
        df (pd.DataFrame): Data to normalize.
        params (Dict[str, Dict[str, float]]): Normalization parameters.

    Returns:
        pd.DataFrame: Normalized data (constant columns become 0).

    Raises:
        ValueError: If the data columns do not match the parameter columns.
    """
    out = df.copy()
    fecha = out["Fecha"] if "Fecha" in out.columns else None
    is_imp = out["is_imputed"] if "is_imputed" in out.columns else None

    data = out.drop(columns=["Fecha", "is_imputed"], errors="ignore").copy()
    # Strict column validation
    cols_data = list(data.columns)
    cols_params = list(params.keys())
    if set(cols_data) != set(cols_params):
        raise ValueError(f"Normalization column mismatch.\nData: {sorted(cols_data)}\nParams: {sorted(cols_params)}")

    for c, p in params.items():
        rng = p["range"]
        if rng == 0:
            data[c] = 0.0
        else:
            data[c] = (data[c] - p["min"]) / rng

    # rebuild
    out = data
    if fecha is not None:
        out.insert(0, "Fecha", fecha.values)
    if is_imp is not None:
        out.insert(1, "is_imputed", is_imp.values)
    return out


# ----------------- BUNDLE WRITER -----------------
@dataclass
class Manifest:
    """Bundle manifest: provenance, column order and file hashes."""
    seed: int
    split: str
    test_year: int
    train_years: List[int]
    target_col: str
    columns_order: List[str]
    hashes: Dict[str, str]


def write_bundle(root_out: str,
                 split: str,
                 test_year: int,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 params: Dict[str, Dict[str, float]]) -> None:
    """Write a bundle to disk.

    Layout::

        <root_out>/{split}/testYYYY/
           train.csv, test.csv, normalization_params.json,
           normalized/train.csv, normalized/test.csv, manifest.json

    Args:
        root_out (str): Bundle root.
        split (str): Split label.
        test_year (int): Test year label.
        train_df (pd.DataFrame): Training partition.
        test_df (pd.DataFrame): Test partition.
        params (Dict[str, Dict[str, float]]): Normalization parameters.
    """
    out_dir = os.path.join(root_out, split, f"test{test_year}")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "normalized"))

    # Save raw (preprocessed) partitions
    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Params
    params_path = os.path.join(out_dir, "normalization_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    # Normalized partitions
    norm_train = normalize_with_params(train_df, params)
    norm_test = normalize_with_params(test_df, params)
    norm_train_path = os.path.join(out_dir, "normalized", "train.csv")
    norm_test_path = os.path.join(out_dir, "normalized", "test.csv")
    norm_train.to_csv(norm_train_path, index=False)
    norm_test.to_csv(norm_test_path, index=False)

    # Manifest
    hashes = {
        "normalization_params.json": sha256_file(params_path),
        "train.csv": sha256_file(train_path),
        "test.csv": sha256_file(test_path),
        "normalized/train.csv": sha256_file(norm_train_path),
        "normalized/test.csv": sha256_file(norm_test_path),
    }
    columns_order = [c for c in train_df.columns if c not in ("Fecha", "is_imputed")]  

    # Derived directly from train_df (not from a heuristic over "all years"), so
    # it is correct in both the leave-one-year-out and the expanding modes.
    train_years = sorted(pd.to_datetime(train_df["Fecha"]).dt.year.unique().tolist())
    manifest = Manifest(
        seed=RANDOM_SEED,
        split=split,
        test_year=test_year,
        train_years=train_years,
        target_col=TARGET_COL,
        columns_order=columns_order,
        hashes=hashes
    )
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Bundle written to {out_dir}")


# ----------------- MAIN PIPELINE -----------------
def detect_full_years(df: pd.DataFrame) -> List[int]:
    """Calendar years fully covered (1 January to 31 December) by the data."""
    years = sorted(df["Fecha"].dt.year.unique().tolist())
    full = []
    for y in years:
        sub = df[(df["Fecha"] >= f"{y}-01-01") & (df["Fecha"] <= f"{y}-12-31")]
        if (not sub.empty) and (sub["Fecha"].min() <= pd.Timestamp(f"{y}-01-01")) and (sub["Fecha"].max() >= pd.Timestamp(f"{y}-12-31")):
            full.append(y)
    return full


def run(config: Any) -> None:
    """Build the leave-one-year-out bundles (``Diciembre`` and ``Junio`` splits).

    Args:
        config (Any): Mapping with ``contextos``.
    """
    df = pd.read_csv(ROOT_RAW)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)

    # Holdout cutoff BEFORE imputing/interpolating: with limit_direction='both',
    # interpolate() could fill a gap on, e.g., 2024-06-28 using an observed
    # holdout value (2024-07-02) as a neighbor; filtering afterwards would leave
    # that leak baked into train. Cutting here removes it at the root.
    df = df[df["Fecha"] < HOLDOUT_CUTOFF_DATE].reset_index(drop=True)

    # Fill NaN cells: every column except the date
    cols_numericas = [c for c in df.columns if c != "Fecha"]

    # Linear interpolation of intermediate gaps; limit_direction='both' also
    # covers NaNs in the first or last row of the dataset.
    df[cols_numericas] = df[cols_numericas].interpolate(method='linear', limit_direction='both')

    df = rellenar_huecos_intermedios_determinista(df, RANDOM_SEED, NOISE_FRAC, TARGET_COL)

    if TEST_YEARS is None:
        full_years = detect_full_years(df)
        # The holdout cutoff already removes 2024/2025 from full_years (incomplete
        # calendar years after HOLDOUT_CUTOFF_DATE), so range(2014, 2026) is a
        # harmless upper bound, not an exclusion that depends on this list.
        candidate = [y for y in range(2014, 2026) if y in full_years]
        test_years = candidate
        if len(test_years) == 0:
            test_years = [y for y in full_years if (y != min(full_years) and y != max(full_years))]
    else:
        test_years = TEST_YEARS

    logger.info(f"Selected test years: {test_years}")

    for split in ("Diciembre", "Junio"):
        for ty in test_years:
            logger.info(f"--- Generating bundle {split} / test{ty} ---")

            if split == "Diciembre":
                start_year = df["Fecha"].min().year
                start_date = pd.Timestamp(f"{start_year}-01-01")
                end_date = pd.Timestamp(f"{start_year}-12-31")
            else:
                start_year = df["Fecha"].min().year
                if df["Fecha"].min().month < 7:
                    start_year = start_year - 1 # if the dataset starts before July, the first hydrological year begins the previous July
                start_date = pd.Timestamp(f"{start_year}-07-01")
                end_date = pd.Timestamp(f"{start_year+1}-06-30")

            df_sub = rellenar_faltantes_primer_subconjunto(
                df=df,
                start_date=start_date,
                end_date=end_date,
                rain_pred_prefixes=RAIN_PRED_PREFIXES,
                target_col=TARGET_COL
            )

            try:
                train_df, test_df = load_and_split_data(df=df_sub, test_year=ty, split=split, context_days=config["contextos"])
            except ValueError as e:
                # The last available year may not have enough future data for a
                # complete "Junio" window; that combination is skipped instead of
                # aborting the rest of the generation.
                logger.warning(f"--- Bundle {split}/test{ty} skipped: {e} ---")
                continue

            params = calculate_normalization_params_excl_imputed(train_df, TARGET_COL)

            write_bundle(ROOT_OUT, split, ty, train_df, test_df, params)


def run_expanding_window(config: Any, test_years: List[int], split: str = "Junio", split_label: str = "JunioExpanding") -> List[int]:
    """Build expanding-window bundles (Train = [dataset start, T-1], Test = hydrological year T).

    Bundles are written under ``split_label`` so that they never overwrite the
    leave-one-year-out ``Diciembre``/``Junio`` bundles.

    Args:
        config (Any): Mapping with ``contextos``.
        test_years (List[int]): Test years.
        split (str): Year convention (``"Junio"`` or ``"Diciembre"``).
        split_label (str): Output split label.

    Returns:
        List[int]: Test years whose bundle was written.
    """
    df = pd.read_csv(ROOT_RAW)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)

    cols_numericas = [c for c in df.columns if c != "Fecha"]
    df[cols_numericas] = df[cols_numericas].interpolate(method="linear", limit_direction="both") # fill gaps by linear interpolation
    df = rellenar_huecos_intermedios_determinista(df, RANDOM_SEED, NOISE_FRAC, TARGET_COL)

    start_year = df["Fecha"].min().year
    if split == "Junio" and df["Fecha"].min().month < 7:
        start_year -= 1
    if split == "Junio":
        start_date = pd.Timestamp(f"{start_year}-07-01")
        end_date = pd.Timestamp(f"{df['Fecha'].max().year}-06-30")
    else:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = pd.Timestamp(f"{df['Fecha'].max().year}-12-31")

    df_sub = rellenar_faltantes_primer_subconjunto(
        df=df, start_date=start_date, end_date=end_date,
        rain_pred_prefixes=RAIN_PRED_PREFIXES, target_col=TARGET_COL,
    )

    written = []
    for ty in test_years:
        logger.info(f"--- Generating expanding bundle {split_label} / test{ty} ---")
        try:
            train_df, test_df = load_and_split_data(
                df=df_sub, test_year=ty, split=split,
                context_days=config["contextos"], expanding_window=True,
            )
        except ValueError as e:
            logger.warning(f"--- Expanding bundle {split_label}/test{ty} skipped: {e} ---")
            continue

        params = calculate_normalization_params_excl_imputed(train_df, TARGET_COL)
        write_bundle(ROOT_OUT, split_label, ty, train_df, test_df, params)
        written.append(ty)

    return written


def run_holdout_bundle(config: Any, holdout_start: str = "2024-07-01", split: str = "Junio",
                        split_label: str = "JunioExpanding", holdout_label: int = 2025) -> Optional[int]:
    """Build the D_dev/D_holdout bundle with an explicit date cutoff.

    ``train.csv`` = D_dev (everything before ``holdout_start``); ``test.csv`` =
    D_holdout (continuous block from ``holdout_start`` to the end of the real
    data, not a whole hydrological year). Written under ``split_label`` next to
    the expanding bundles; ``holdout_label`` names its directory.

    Args:
        config (Any): Mapping with ``contextos``.
        holdout_start (str): First holdout date.
        split (str): Year convention.
        split_label (str): Output split label.
        holdout_label (int): Directory label of the holdout bundle.

    Returns:
        Optional[int]: ``holdout_label`` if written, else ``None``.
    """
    df = pd.read_csv(ROOT_RAW)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)

    cols_numericas = [c for c in df.columns if c != "Fecha"]
    df[cols_numericas] = df[cols_numericas].interpolate(method="linear", limit_direction="both")
    df = rellenar_huecos_intermedios_determinista(df, RANDOM_SEED, NOISE_FRAC, TARGET_COL)

    start_year = df["Fecha"].min().year
    if split == "Junio" and df["Fecha"].min().month < 7:
        start_year -= 1
    start_date = pd.Timestamp(f"{start_year}-07-01") if split == "Junio" else pd.Timestamp(f"{start_year}-01-01")
    end_date = df["Fecha"].max()  # D_holdout is dynamic: up to the last real data point.

    df_sub = rellenar_faltantes_primer_subconjunto(
        df=df, start_date=start_date, end_date=end_date,
        rain_pred_prefixes=RAIN_PRED_PREFIXES, target_col=TARGET_COL,
    )

    eval_start = pd.Timestamp(holdout_start)
    eval_end = df_sub["Fecha"].max()
    try:
        train_df, test_df = load_and_split_data(
            df=df_sub, test_year=holdout_label, split=split,
            context_days=config["contextos"], expanding_window=True,
            eval_range=(eval_start, eval_end),
        )
    except ValueError as e:
        logger.warning(f"--- Holdout bundle {split_label}/test{holdout_label} skipped: {e} ---")
        return None

    params = calculate_normalization_params_excl_imputed(train_df, TARGET_COL)
    write_bundle(ROOT_OUT, split_label, holdout_label, train_df, test_df, params)
    return holdout_label


if __name__ == "__main__":
    run(CONFIG)