"""Tests for resultados/prueba63/src/data/merge_datasets.py.

- Exactly 9 columns (historical + new, without 'pred_l/m2_4d').
- Continuous dates, no gaps longer than 1 day.
- No duplicate dates.
- Correct overlap handling (the first occurrence is kept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resultados.prueba63.src.data.merge_datasets import merge_datasets, STANDARD_COLUMNS


def _write_historic(path, start="2014-01-01", periods=20):
    dates = pd.date_range(start, periods=periods, freq="D")
    df = pd.DataFrame({
        "Fecha": dates,
        "Qe": np.linspace(50, 70, periods),
        "l/m2_arzua": np.full(periods, 1.0),
        "l/m2_serradofaro": np.full(periods, 2.0),
        "l/m2_melide": np.full(periods, 3.0),
        "l/m2_olveda": np.full(periods, 4.0),
        "pred_l/m2": np.full(periods, 5.0),
        "pred_l/m2_2d": np.full(periods, 6.0),
        "pred_l/m2_3d": np.full(periods, 7.0),
    })
    df.to_csv(path, index=False)
    return df


def _write_new(path, start, periods=15, with_pred_4d=True):
    dates = pd.date_range(start, periods=periods, freq="D")
    df = pd.DataFrame({
        "Fecha": dates,
        "Qe": np.linspace(80, 100, periods),
        "l/m2_arzua": np.full(periods, 10.0),
        "l/m2_serradofaro": np.full(periods, 20.0),
        "l/m2_melide": np.full(periods, 30.0),
        "l/m2_olveda": np.full(periods, 40.0),
        "pred_l/m2": np.full(periods, 50.0),
        "pred_l/m2_2d": np.full(periods, 60.0),
        "pred_l/m2_3d": np.full(periods, 70.0),
    })
    if with_pred_4d:
        df["pred_l/m2_4d"] = np.full(periods, 999.0)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------- #
# Overlapping case (10 days shared by the historical and new records)
# --------------------------------------------------------------------------- #

@pytest.fixture
def merged_with_overlap(tmp_path):
    historic_path = tmp_path / "historic.csv"
    new_path = tmp_path / "new.csv"
    out_path = tmp_path / "merged.csv"

    # historical: 2014-01-01 .. 2014-01-20; new: 2014-01-11 .. 2014-01-25 (10-day overlap)
    _write_historic(historic_path, start="2014-01-01", periods=20)
    _write_new(new_path, start="2014-01-11", periods=15, with_pred_4d=True)

    merged = merge_datasets(str(historic_path), str(new_path), str(out_path))
    return merged, out_path


def test_merge_has_exact_nine_standard_columns(merged_with_overlap):
    merged, _ = merged_with_overlap
    assert list(merged.columns) == STANDARD_COLUMNS
    assert "pred_l/m2_4d" not in merged.columns


def test_merge_dates_are_continuous_no_gaps(merged_with_overlap):
    merged, _ = merged_with_overlap
    dates = pd.to_datetime(merged["Fecha"])
    gaps = dates.diff().dropna()
    assert (gaps == pd.Timedelta(days=1)).all()


def test_merge_no_duplicate_dates(merged_with_overlap):
    merged, _ = merged_with_overlap
    assert merged["Fecha"].duplicated().sum() == 0
    assert merged["Fecha"].is_unique


def test_merge_keeps_first_occurrence_on_overlap(merged_with_overlap):
    """In the overlap (2014-01-11..2014-01-20) the historical value must win over the new one (keep='first')."""
    merged, _ = merged_with_overlap
    merged["Fecha"] = pd.to_datetime(merged["Fecha"])
    overlap_row = merged[merged["Fecha"] == "2014-01-11"].iloc[0]
    # historical l/m2_arzua=1.0, new 10.0
    assert overlap_row["l/m2_arzua"] == pytest.approx(1.0)


def test_merge_row_count_accounts_for_overlap(merged_with_overlap):
    merged, _ = merged_with_overlap
    # historical 20 days (2014-01-01..20) + new 15 days (2014-01-11..25),
    # 10-day overlap -> 20 + 15 - 10 = 25 unique days.
    assert len(merged) == 25


def test_merge_writes_output_file(merged_with_overlap):
    _, out_path = merged_with_overlap
    assert out_path.exists()
    on_disk = pd.read_csv(out_path)
    assert list(on_disk.columns) == STANDARD_COLUMNS


# --------------------------------------------------------------------------- #
# No overlap but a date gap between both records: it is interpolated
# --------------------------------------------------------------------------- #

def test_merge_interpolates_gap_between_non_overlapping_ranges(tmp_path):
    historic_path = tmp_path / "historic.csv"
    new_path = tmp_path / "new.csv"

    # historical ends 2014-01-10, new starts 2014-01-15 -> 4-day gap
    _write_historic(historic_path, start="2014-01-01", periods=10)
    _write_new(new_path, start="2014-01-15", periods=5, with_pred_4d=False)

    merged = merge_datasets(str(historic_path), str(new_path), output_path=None)
    merged["Fecha"] = pd.to_datetime(merged["Fecha"])

    dates = merged["Fecha"]
    gaps = dates.diff().dropna()
    assert (gaps == pd.Timedelta(days=1)).all()  # no gaps after interpolation
    assert not merged["Qe"].isna().any()  # intermediate NaNs interpolated, none left


def test_merge_missing_column_raises(tmp_path):
    historic_path = tmp_path / "historic.csv"
    new_path = tmp_path / "new.csv"
    _write_historic(historic_path)

    bad = pd.DataFrame({"Fecha": pd.date_range("2024-01-01", periods=5), "Qe": range(5)})
    bad.to_csv(new_path, index=False)

    with pytest.raises(ValueError):
        merge_datasets(str(historic_path), str(new_path), output_path=None)


# --------------------------------------------------------------------------- #
# Integration with the real project CSVs (when available)
# --------------------------------------------------------------------------- #

def test_merge_real_project_datasets_end_to_end(tmp_path):
    from resultados.prueba63.src.data.merge_datasets import DEFAULT_HISTORIC_PATH, DEFAULT_NEW_PATH
    import os

    if not (os.path.exists(DEFAULT_HISTORIC_PATH) and os.path.exists(DEFAULT_NEW_PATH)):
        pytest.skip("Real project CSVs not available in this environment.")

    out_path = tmp_path / "dataset_completo_real.csv"
    merged = merge_datasets(DEFAULT_HISTORIC_PATH, DEFAULT_NEW_PATH, str(out_path))

    assert list(merged.columns) == STANDARD_COLUMNS
    dates = pd.to_datetime(merged["Fecha"])
    assert (dates.diff().dropna() == pd.Timedelta(days=1)).all()
    assert merged["Fecha"].duplicated().sum() == 0
    assert not merged.isna().any().any()
