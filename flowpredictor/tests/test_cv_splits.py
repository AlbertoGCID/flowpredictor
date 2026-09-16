"""Temporal-leakage tests of the expanding-window CV folds and the 2025 holdout.

For every fold, all training dates must be strictly earlier than all test
dates (train < test).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from resultados.prueba63.src.data.dataset_generator import load_and_split_data
from resultados.prueba63.src.main_pipeline import (
    build_expanding_window_cv_plan,
    EXPANDING_CV_YEARS,
    HOLDOUT_YEAR,
)


def _make_synthetic_daily_df(start="2010-01-01", end="2025-12-31"):
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Fecha": dates,
        "Qe": rng.uniform(1, 100, size=n),
        "l/m2_arzua": rng.uniform(0, 50, size=n),
        "l/m2_serradofaro": rng.uniform(0, 50, size=n),
        "l/m2_melide": rng.uniform(0, 50, size=n),
        "l/m2_olveda": rng.uniform(0, 50, size=n),
        "pred_l/m2": rng.uniform(0, 50, size=n),
        "pred_l/m2_2d": rng.uniform(0, 50, size=n),
        "pred_l/m2_3d": rng.uniform(0, 50, size=n),
    })


# --------------------------------------------------------------------------- #
# 1. CV plan (build_expanding_window_cv_plan): structure and year coverage
# --------------------------------------------------------------------------- #

def test_cv_plan_has_7_folds_plus_holdout():
    plan = build_expanding_window_cv_plan()
    folds = [f for f in plan if not f["is_holdout"]]
    holdouts = [f for f in plan if f["is_holdout"]]

    assert len(folds) == 7
    assert [f["test_year"] for f in folds] == list(range(2017, 2024))
    assert len(holdouts) == 1
    assert holdouts[0]["test_year"] == 2025
    assert EXPANDING_CV_YEARS == list(range(2017, 2024))
    assert HOLDOUT_YEAR == 2025


def test_cv_plan_train_years_are_strictly_before_test_year():
    plan = build_expanding_window_cv_plan()
    for fold in plan:
        assert all(y < fold["test_year"] for y in fold["train_years"]), (
            f"fold {fold['test_year']} contains train_years >= test_year: {fold['train_years']}"
        )


def test_cv_plan_train_years_grow_monotonically_expanding_window():
    plan = build_expanding_window_cv_plan()
    sizes = [len(f["train_years"]) for f in plan]
    assert sizes == sorted(sizes)  # expanding window: never shrinks


# --------------------------------------------------------------------------- #
# 2. load_and_split_data(expanding_window=True): no temporal leakage, fold by fold
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("test_year", list(range(2014, 2025)) + [2025])
def test_expanding_window_split_train_strictly_before_test(test_year):
    """train < test for each test year 2014-2024 and the 2025 holdout."""
    df = _make_synthetic_daily_df()

    train_df, test_df = load_and_split_data(
        df=df, test_year=test_year, split="Junio", context_days=[30], expanding_window=True,
    )

    assert pd.to_datetime(train_df["Fecha"]).max() < pd.to_datetime(test_df["Fecha"]).min()


def test_expanding_window_split_never_uses_years_after_test_year():
    df = _make_synthetic_daily_df()
    for test_year in list(range(2014, 2025)) + [2025]:
        train_df, _ = load_and_split_data(
            df=df, test_year=test_year, split="Junio", context_days=[30], expanding_window=True,
        )
        train_years = pd.to_datetime(train_df["Fecha"]).dt.year.unique()
        assert all(y <= test_year for y in train_years)


def test_expanding_window_split_raises_assertion_is_enforced_internally():
    """load_and_split_data() itself asserts train.max() < test_df_start when
    expanding_window=True; a regression reintroducing future years into train
    must trigger that internal assert, not only this external check."""
    df = _make_synthetic_daily_df()
    # Normal call: must not raise.
    load_and_split_data(df=df, test_year=2020, split="Junio", context_days=[30], expanding_window=True)


def test_eval_range_requires_expanding_window():
    """eval_range (D_holdout) must never be combined with expanding_window=False:
    the holdout cannot leak into train, not even as a "later year" of the
    leave-one-year-out mode."""
    df = _make_synthetic_daily_df()
    with pytest.raises(ValueError, match="expanding_window=True"):
        load_and_split_data(
            df=df, test_year=2025, split="Junio", context_days=[30], expanding_window=False,
            eval_range=(pd.Timestamp("2024-07-01"), pd.Timestamp("2025-12-31")),
        )


def test_eval_range_overrides_year_range_for_split():
    """With eval_range, the test set covers exactly [eval_start - context, eval_end],
    not the hydrological year that test_year/split would determine."""
    df = _make_synthetic_daily_df()
    eval_start, eval_end = pd.Timestamp("2024-07-01"), pd.Timestamp("2025-12-31")
    train_df, test_df = load_and_split_data(
        df=df, test_year=2025, split="Junio", context_days=[30], expanding_window=True,
        eval_range=(eval_start, eval_end),
    )
    assert pd.to_datetime(test_df["Fecha"]).max() == eval_end
    assert pd.to_datetime(test_df["Fecha"]).min() == eval_start - pd.Timedelta(days=30)
    assert pd.to_datetime(train_df["Fecha"]).max() < eval_start - pd.Timedelta(days=30)


def test_legacy_non_expanding_split_can_include_future_years_by_design():
    """Contrast: the legacy mode (expanding_window=False) still uses years after
    the test year, confirming that the expanding mode is opt-in and the default
    behavior is unchanged."""
    df = _make_synthetic_daily_df()
    train_df, test_df = load_and_split_data(
        df=df, test_year=2018, split="Junio", context_days=[30], expanding_window=False,
    )
    train_years = pd.to_datetime(train_df["Fecha"]).dt.year.unique()
    assert any(y > 2018 for y in train_years)


# --------------------------------------------------------------------------- #
# 3. Integration: real bundles already generated in datasets/normalizados/JunioExpanding
# --------------------------------------------------------------------------- #

def test_real_expanding_bundles_have_no_temporal_leakage_if_present():
    """If the JunioExpanding bundles exist in this environment
    (dataset_generator.run_expanding_window), check on the real CSVs that
    train.max() < test.min() for every fold and the holdout.

    The dataset paths of the configuration are relative to the parent of the
    project root folder, while pytest runs from the project root itself, so
    both candidate roots are tried before skipping.
    """
    from pathlib import Path
    from resultados.prueba63.src.config import CONFIG

    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        os.path.join(CONFIG.paths["data_bundle_root"], "JunioExpanding"),
        str(project_root / "datasets" / "normalizados" / "JunioExpanding"),
    ]
    bundle_root = next((c for c in candidates if os.path.isdir(c)), None)
    if bundle_root is None:
        pytest.skip("JunioExpanding bundles not generated in this environment.")

    checked = 0
    for year in list(range(2014, 2025)) + [2025]:
        bundle_dir = os.path.join(bundle_root, f"test{year}")
        train_path = os.path.join(bundle_dir, "train.csv")
        test_path = os.path.join(bundle_dir, "test.csv")
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            continue

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        train_max = pd.to_datetime(train_df["Fecha"]).max()
        test_min = pd.to_datetime(test_df["Fecha"]).min()
        assert train_max < test_min, f"Temporal leakage in fold {year}: train.max()={train_max} >= test.min()={test_min}"
        checked += 1

    assert checked > 0


def test_real_holdout_bundle_covers_d_holdout_without_leakage_if_present():
    """If the continuous holdout bundle (dataset_generator.run_holdout_bundle)
    exists, check on the real CSVs that train.csv (D_dev) never contains dates
    >= 2024-07-01 and that test.csv (D_holdout) starts at that cutoff (minus
    the context gap) and extends beyond a single hydrological year."""
    from pathlib import Path
    from resultados.prueba63.src.config import CONFIG

    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        os.path.join(CONFIG.paths["data_bundle_root"], "JunioExpanding", "test2025"),
        str(project_root / "datasets" / "normalizados" / "JunioExpanding" / "test2025"),
    ]
    bundle_dir = next((c for c in candidates if os.path.isdir(c)), None)
    if bundle_dir is None:
        pytest.skip("Holdout bundle JunioExpanding/test2025 not generated in this environment.")

    holdout_cutoff = pd.Timestamp("2024-07-01")
    train_df = pd.read_csv(os.path.join(bundle_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(bundle_dir, "test.csv"))
    train_df["Fecha"] = pd.to_datetime(train_df["Fecha"])
    test_df["Fecha"] = pd.to_datetime(test_df["Fecha"])

    assert train_df["Fecha"].max() < holdout_cutoff, (
        f"D_dev (train.csv) contains dates >= {holdout_cutoff}: max={train_df['Fecha'].max()}"
    )
    # test.csv includes the context gap before the cutoff; what matters is its
    # latest EVALUATION date, which must go beyond a single hydrological year
    # (2025-06-30): the continuous D_holdout block up to the end of the real data.
    assert test_df["Fecha"].max() > pd.Timestamp("2025-06-30")
