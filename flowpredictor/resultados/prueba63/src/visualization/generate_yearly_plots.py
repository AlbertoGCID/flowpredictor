"""Yearly hydrograph figure per (configuration, year) from artifacts already on disk.

Never trains nor loads any Keras/sklearn model.

Data sources: the flat ``.npz`` files in ``resultados/prueba63/predictions/``
(one per fold, ``y_true``/``y_pred`` only) list WHICH combinations exist, but
carry neither dates nor the per-penalty breakdown; those live in the per-hash
cache ``tests63/<hash>/predictions_cache/``. This script maps each fold to its
hash through ``results_consolidated.csv`` and plots:

- Observed inflow (``y_true``) with red markers above the TRAINING p90
  threshold (read from ``metrics_<tag>.json`` -> ``top10.threshold``, the same
  value used for the fold metrics; never recomputed on test).
- Baseline: always M1 (symmetric MAE, no pretraining), the common reference
  in every figure. M1 itself has no baseline.
- Penalized band: min/max envelope of the pinball variants (penalty > 0) of
  the configuration itself (M5). Single-variant configurations have no band.
- RF classifier prediction (M5 only, ``classifier_<algo>.npz``).
"""

from __future__ import annotations

# Absolute imports (from resultados.prueba63...): run with flowpredictor/ on
# PYTHONPATH, either as a module (-m) or as a script.

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resultados.prueba63.src.config_schema import CONFIG
from resultados.prueba63.src.main_pipeline import ABLATION_CONFIGS, HOLDOUT_YEAR
from resultados.prueba63.src.evaluation.model_selection import filter_published_variant

# Paths: experiments_config.toml [paths].
PREDICTIONS_DIR = CONFIG.paths["predictions_dir"]
CACHE_ROOT = CONFIG.paths["cache_root"]
RESULTS_CSV = CONFIG.paths["results_csv"]
OUT_DIR = CONFIG.paths["yearly_plots_dir"]

_ABLATION_BY_NAME = {c["name"]: c for c in ABLATION_CONFIGS}


def _baseline_sibling(config: str) -> str | None:
    """Baseline configuration of a figure: always M1, the common reference.

    Args:
        config (str): Plotted configuration.

    Returns:
        str | None: ``"M1"``, or ``None`` for M1 itself.
    """
    return None if config == "M1" else "M1"


def _load_npz(path: str) -> dict:
    """Load every array of an ``.npz`` file into a dict."""
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.keys()}


def _own_model_variants(cache_dir: str) -> dict:
    """Per-penalty test predictions of a cache directory.

    Args:
        cache_dir (str): ``predictions_cache`` directory.

    Returns:
        dict: ``{penalty: {dates, y_true, y_pred, ...}}`` from ``pred_*.npz``
        (``*_train.npz`` and ``classifier_*.npz`` excluded).
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(cache_dir, "pred_*.npz"))):
        base = os.path.basename(p)
        if base.endswith("_train.npz"):
            continue
        m = re.search(r"_(\d+)\.npz$", base)
        if not m:
            continue
        out[int(m.group(1))] = _load_npz(p)
    return out


def _classifier_variant(cache_dir: str) -> dict | None:
    """Meta-learner predictions of a cache directory, if any."""
    matches = [p for p in glob.glob(os.path.join(cache_dir, "classifier_*.npz"))]
    return _load_npz(matches[0]) if matches else None


def _train_threshold_p90(cache_dir: str) -> float | None:
    """Training p90 threshold stored in the cached ``metrics_*.json`` files, if any."""
    for p in sorted(glob.glob(os.path.join(cache_dir, "metrics_*.json"))):
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        thr = m.get("top10", {}).get("threshold")
        if thr is not None:
            return float(thr)
    return None


def plot_year(config: str, year: int, own_hash: str, results_df: pd.DataFrame, out_dir: str,
              suffix: str = "", strategy: str = "expanding") -> str | None:
    """Plot and save the yearly hydrograph of one configuration.

    Args:
        config (str): Configuration (``M1``-``M5``).
        year (int): Test year.
        own_hash (str): Iteration hash of the fold.
        results_df (pd.DataFrame): Published-variant results (for the M1 lookup).
        out_dir (str): Output directory.
        suffix (str): File-name suffix (e.g. ``_loyo``).
        strategy (str): Strategy used to find the matching M1 fold.

    Returns:
        str | None: Path of the PNG, or ``None`` if the cache is missing.
    """
    cache_dir = os.path.join(CACHE_ROOT, own_hash, "predictions_cache")
    if not os.path.isdir(cache_dir):
        print(f"[yearly_plots] {config}_{year}: no predictions_cache ({cache_dir}); skipping.")
        return None

    variants = _own_model_variants(cache_dir)
    if not variants:
        print(f"[yearly_plots] {config}_{year}: no pred_*.npz in the cache; skipping.")
        return None

    any_variant = next(iter(variants.values()))
    dates = pd.to_datetime(any_variant["dates"])
    y_true = any_variant["y_true"].astype(float)
    threshold = _train_threshold_p90(cache_dir)

    penalties = sorted(variants.keys())
    is_band_config = len(penalties) > 1  # M2/M4/M5: penalty in {2,4,6,8,10}

    baseline = None
    baseline_cfg = _baseline_sibling(config)
    if baseline_cfg is not None:
        sib_rows = results_df[(results_df["config"] == baseline_cfg) & (results_df["test_year"] == year)]
        if "strategy" in results_df.columns:
            sib_rows = sib_rows[sib_rows["strategy"] == strategy]
        if not sib_rows.empty:
            sib_hash = sib_rows.iloc[0]["hash"]
            sib_cache = os.path.join(CACHE_ROOT, sib_hash, "predictions_cache")
            sib_npz = glob.glob(os.path.join(sib_cache, "pred_original_mae_0.npz"))
            if sib_npz:
                baseline = _load_npz(sib_npz[0])

    classifier = _classifier_variant(cache_dir) if _ABLATION_BY_NAME[config]["use_rf"] else None

    # --- style ---
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "legend.fontsize": 11, "axes.titlesize": 15,
        "figure.dpi": 100,
    })
    plt.figure(figsize=(14, 4.5))
    plt.plot(dates, y_true, linewidth=1.4, alpha=0.85, linestyle="-", color="black", label="Observed")

    if baseline is not None:
        b_dates = pd.to_datetime(baseline["dates"])
        if len(b_dates) == len(dates) and np.all(b_dates.values == dates.values):
            plt.plot(dates, baseline["y_pred"].astype(float), linewidth=1.6, alpha=0.9,
                      linestyle=":", color="tab:blue", label=f"Baseline ({baseline_cfg})")
        else:
            print(f"[yearly_plots] {config}_{year}: baseline dates ({baseline_cfg}) are not aligned; omitted.")

    if is_band_config:
        stack = np.stack([variants[p]["y_pred"].astype(float) for p in penalties], axis=0)
        lo, hi = stack.min(axis=0), stack.max(axis=0)
        plt.fill_between(dates, lo, hi, alpha=0.25, color="#9ecae1",
                          label=f"Penalized band (penalty {penalties[0]}-{penalties[-1]})")
        plt.plot(dates, stack.mean(axis=0), linewidth=1.6, alpha=0.9, color="#9ecae1",
                  label="Penalized mean")
    else:
        # M1/M3 (penalty=0) and M2/M4 (single penalty=2) have a single
        # trained variant -- no band to draw.
        only = variants[penalties[0]]
        plt.plot(dates, only["y_pred"].astype(float), linewidth=1.8, alpha=0.95,
                  color="#9ecae1", label=f"Model (penalty {penalties[0]})")

    if classifier is not None:
        c_dates = pd.to_datetime(classifier["dates"])
        if len(c_dates) == len(dates) and np.all(c_dates.values == dates.values):
            plt.plot(dates, classifier["y_pred"].astype(float), linewidth=1.8, alpha=0.95,
                      color="tab:orange", label="Classifier prediction")
        else:
            print(f"[yearly_plots] {config}_{year}: classifier dates are not aligned; omitted.")

    if threshold is not None:
        plt.axhline(threshold, color="black", linewidth=1.0, alpha=0.3, linestyle=":",
                     label="P90 threshold (train)")
        mask_top = y_true >= threshold
        if np.any(mask_top):
            plt.scatter(dates[mask_top], y_true[mask_top], s=22, marker="o", facecolor="none",
                         edgecolor="red", linewidths=1.1, alpha=0.9, label="Observed extremes")
    else:
        print(f"[yearly_plots] {config}_{year}: no p90 threshold in metrics_*.json; threshold line and markers omitted.")

    # Fixed title format that always shows the four fields (including
    # "Meta-Learner: None" when it does not apply).
    cfg_meta = _ABLATION_BY_NAME[config]
    if cfg_meta["loss_name"] == "original_mae":
        loss_desc = "MAE"
    elif is_band_config:
        loss_desc = f"Pinball p={penalties[0]}-{penalties[-1]}"
    else:
        loss_desc = f"Pinball p={penalties[0]}"
    meta_learner = "RF" if classifier is not None else "None"
    plt.title(f"Inflow Forecast {year} - {config} - Loss: {loss_desc} - Meta-Learner: {meta_learner}")
    plt.ylabel("Inflow (m³/s)")
    plt.xlabel("Date")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.grid(alpha=0.25, linewidth=0.8)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{config}_{year}{suffix}"
    png = os.path.join(out_dir, f"{fname}.png")
    pdf = os.path.join(out_dir, f"{fname}.pdf")
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()
    return png


def main() -> None:
    """Plot every M1-M5 fold with saved predictions."""
    if not os.path.exists(RESULTS_CSV):
        print(f"[yearly_plots] {RESULTS_CSV} does not exist; nothing to plot.")
        return
    # Published variant only: sensitivity rows (p95, other seeds) share
    # config/test_year/strategy, so a hash lookup by those keys could return
    # another variant's row.
    results_df = filter_published_variant(pd.read_csv(RESULTS_CSV))

    # Fold labels are "M{1-5}_{year}" (expanding), "M{1-5}_loyo_{year}" and
    # "M{1-5}_holdout_final" (no year in the name, test_year=HOLDOUT_YEAR). Each
    # pattern carries its own strategy so that a fold's hash is never confused
    # with that of another strategy for the same (config, year).
    _FOLD_PATTERNS = (
        (re.compile(r"^(M[1-5])_(\d{4})$"), "expanding"),
        (re.compile(r"^(M[1-5])_loyo_(\d{4})$"), "loyo"),
        (re.compile(r"^(M[1-5])_holdout_final$"), "holdout_final"),
    )

    fold_files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*.npz")))
    saved = []
    for path in fold_files:
        fold = os.path.splitext(os.path.basename(path))[0]
        config = year = strategy = None
        for pattern, strat in _FOLD_PATTERNS:
            m = pattern.match(fold)
            if m:
                config = m.group(1)
                year = int(m.group(2)) if strat != "holdout_final" else HOLDOUT_YEAR
                strategy = strat
                break
        if config is None:
            continue  # baselines/heuristic_ensemble/robustness: out of scope for this figure

        row = results_df[(results_df["config"] == config) & (results_df["test_year"] == year)]
        if "strategy" in results_df.columns:
            row = row[row["strategy"] == strategy]
        if row.empty:
            print(f"[yearly_plots] {fold}: no row in results_consolidated.csv; skipping.")
            continue
        own_hash = row.iloc[0]["hash"]

        suffix = "" if strategy == "expanding" else f"_{strategy}"
        out = plot_year(config, year, own_hash, results_df, OUT_DIR, suffix=suffix, strategy=strategy)
        if out:
            saved.append(out)

    print(f"[yearly_plots] {len(saved)} figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
