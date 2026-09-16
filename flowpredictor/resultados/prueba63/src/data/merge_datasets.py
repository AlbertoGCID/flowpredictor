"""Merge the historical record (up to 2023-12-31) with the 2024-2025 extension.

Produces a single continuous daily CSV aligned to the 9 standard columns.

Note: despite its file name (``dataset_completo_2009_2025.csv``), the
historical record starts on 2014-01-01, not in 2009; the merge never
fabricates data before that date, it only concatenates what exists.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

STANDARD_COLUMNS: List[str] = [
    "Fecha", "Qe", "l/m2_arzua", "l/m2_serradofaro", "l/m2_melide",
    "l/m2_olveda", "pred_l/m2", "pred_l/m2_2d", "pred_l/m2_3d",
]

# <project root>/resultados/prueba63/src/data/merge_datasets.py -> parents[4] == <project root>/
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HISTORIC_PATH = str(_PROJECT_ROOT / "datasets" / "dataframe_original.csv")
DEFAULT_NEW_PATH = str(_PROJECT_ROOT / "datasets" / "dataset_2024_2025.csv")
DEFAULT_OUTPUT_PATH = str(_PROJECT_ROOT / "datasets" / "dataset_completo_2009_2025.csv")


def _load_and_align(path: str) -> pd.DataFrame:
    """Read a source CSV, drop ``pred_l/m2_4d`` if present, and validate/order the 9 standard columns.

    Args:
        path (str): CSV path.

    Returns:
        pd.DataFrame: Aligned data with a datetime ``Fecha``.

    Raises:
        ValueError: If a standard column is missing.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["pred_l/m2_4d"], errors="ignore")

    missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[merge_datasets] {path} is missing the expected columns: {missing}")

    df = df[STANDARD_COLUMNS].copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df


def merge_datasets(
    historic_path: str = DEFAULT_HISTORIC_PATH,
    new_path: str = DEFAULT_NEW_PATH,
    output_path: Optional[str] = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Merge the historical and new records into one continuous daily series.

    1. Align both to the 9 standard columns (dropping ``pred_l/m2_4d``).
    2. Concatenate, drop duplicate dates (keeping the first occurrence, so the
       historical record wins in the overlap) and sort chronologically.
    3. Reindex to a continuous daily calendar and linearly interpolate any
       remaining NaNs (date gaps or empty cells) in every column but ``Fecha``.

    Args:
        historic_path (str): Historical CSV.
        new_path (str): New-period CSV.
        output_path (Optional[str]): Output CSV; ``None`` skips writing (tests).

    Returns:
        pd.DataFrame: Merged data.
    """
    historic = _load_and_align(historic_path)
    new = _load_and_align(new_path)

    merged = pd.concat([historic, new], ignore_index=True)
    merged = merged.sort_values("Fecha")
    merged = merged.drop_duplicates(subset="Fecha", keep="first")
    merged = merged.sort_values("Fecha").reset_index(drop=True)

    full_range = pd.date_range(merged["Fecha"].min(), merged["Fecha"].max(), freq="D")
    merged = merged.set_index("Fecha").reindex(full_range)
    merged.index.name = "Fecha"

    value_cols = [c for c in STANDARD_COLUMNS if c != "Fecha"]
    merged[value_cols] = merged[value_cols].interpolate(method="linear", limit_direction="both")

    merged = merged.reset_index()[STANDARD_COLUMNS]

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        merged.to_csv(output_path, index=False)

    return merged


if __name__ == "__main__":
    out = merge_datasets()
    print(
        f"[merge_datasets] {len(out)} rows saved to {DEFAULT_OUTPUT_PATH} "
        f"({out['Fecha'].min().date()} -> {out['Fecha'].max().date()})"
    )
