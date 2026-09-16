#!/usr/bin/env python3
"""Paper figures (matplotlib/seaborn, 300 DPI, PDF+EPS+PNG) and LaTeX tables (booktabs).

Generated exclusively from saved artifacts, never by retraining:
  - resultados/prueba63/results_consolidated.csv (run_experiments_phase3.py)
  - resultados/prueba63/predictions/robustness_*.npz
  - the per-hash prediction caches (tests63/<hash>/predictions_cache/)

Uses absolute imports and paths from experiments_config.toml: run it with
flowpredictor/ on PYTHONPATH, as a module (-m) or as a script.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from resultados.prueba63.src.config_schema import CONFIG
from resultados.prueba63.src.evaluation.model_selection import filter_published_variant, run_model_selection
from resultados.prueba63.src.main_pipeline import ABLATION_CONFIGS

# Paths and reporting constants: experiments_config.toml [paths]/[reporting].
RESULTS_CSV = Path(CONFIG.paths["results_csv"])
PREDICTIONS_DIR = Path(CONFIG.paths["predictions_dir"])
DOC_DIR = Path(CONFIG.paths["doc_dir"])
DPI = int(CONFIG.reporting["dpi"])
FIGSIZE_WIDE = tuple(CONFIG.reporting["figsize_wide"])

sns.set_theme(style="whitegrid", context="paper")

ABLATION_ORDER = list(CONFIG.reporting["ablation_order"])
BASELINE_ORDER = list(CONFIG.reporting["baseline_order"])
# Expanding-window CV range: test years 2017-2023, never mixed with the holdout
# (2024-07-01 onwards) nor with the burn-in years (2014-2016).
EXPANDING_CV_YEAR_RANGE = tuple(CONFIG.reporting["expanding_cv_year_range"])


def _best_config(results_csv: Path = RESULTS_CSV) -> str:
    """Configuration M* selected by the model-selection rule.

    Paper tables and figures follow the objectively selected configuration,
    not a hard-coded one.

    Args:
        results_csv (Path): Consolidated results.

    Returns:
        str: The selected configuration, or ``"M5"`` if selection fails.
    """
    try:
        return run_model_selection(str(results_csv))["selected"]
    except Exception as e:
        print(f"[generate_paper_figures] Could not determine M* ({e}); falling back to M5.")
        return "M5"


def _filter_horizon(df: pd.DataFrame, horizon: str = "48h") -> pd.DataFrame:
    """Keep a single forecast horizon BEFORE aggregating or plotting anything.

    Without it, 24h/48h/72h rows of the same (config, test_year, strategy)
    would be silently averaged together. A ``df`` without a ``horizon``
    column (older CSVs, synthetic tests) is returned unfiltered.

    Args:
        df (pd.DataFrame): Results.
        horizon (str): Horizon to keep.

    Returns:
        pd.DataFrame: Filtered results.
    """
    if "horizon" not in df.columns:
        return df
    return df[df["horizon"] == horizon]


def _save_fig(fig: plt.Figure, name: str, doc_dir: Path) -> None:
    """Save a figure as PDF, EPS and PNG and close it.

    Args:
        fig (plt.Figure): Figure.
        name (str): Base file name.
        doc_dir (Path): Output directory.
    """
    doc_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(doc_dir / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[generate_paper_figures] {name}: saved to {doc_dir} (pdf/eps/png, {DPI} dpi)")


def load_results(results_csv: Path = RESULTS_CSV) -> Optional[pd.DataFrame]:
    """Load the published-variant rows of the consolidated results.

    Args:
        results_csv (Path): Consolidated results.

    Returns:
        Optional[pd.DataFrame]: Results, or ``None`` if missing or empty.
    """
    if not results_csv.exists():
        print(f"[generate_paper_figures] {results_csv} does not exist; run run_experiments_phase3.py first.")
        return None
    # Manuscript figures and tables ALWAYS use the published variant (seed 92,
    # p90). Sensitivity variants share config/strategy/horizon and would be
    # mixed into means and .iloc[0] lookups.
    df = filter_published_variant(pd.read_csv(results_csv))
    if df.empty:
        print(f"[generate_paper_figures] {results_csv} is empty.")
        return None
    return df

# --------------------------------------------------------------------------- #
# Figure_MultiYear: year-over-year evolution (CV + holdout), 95% CI
# --------------------------------------------------------------------------- #

def figure_multiyear(df: pd.DataFrame, metric: str = "NSE", doc_dir: Path = DOC_DIR) -> Optional[Path]:
    """Two-panel year-over-year evolution (expanding CV + holdout).

    Top panel NSE and KGE, bottom panel Hit Ratio and FARate, for M* compared
    with M1 and the baselines.

    Args:
        df (pd.DataFrame): Published-variant results.
        metric (str): Kept for CLI signature compatibility; the four metrics
            are always plotted.
        doc_dir (Path): Output directory.

    Returns:
        Optional[Path]: Base path of the saved figure, or ``None``.
    """
    best_config = _best_config()
    configs = _table_configs(best_config)  # [M*, 3 baselines, M1]

    df = _filter_horizon(df)
    rows = df[df["strategy"].isin(["expanding", "holdout_final"])] if "strategy" in df.columns else df
    rows = rows[rows["config"].isin(configs)]
    if rows.empty:
        print("[Figure_MultiYear] No rows for M*/baselines/M1; skipped.")
        return None

    colors = dict(zip(configs, sns.color_palette("tab10", n_colors=len(configs))))
    holdout_years = sorted(rows.loc[rows.get("strategy") == "holdout_final", "test_year"].unique()) \
        if "strategy" in rows.columns else []

    def _plot_series(ax, metric_name: str, marker: str, linestyle: str, with_ci: bool, with_label: bool):
        if metric_name not in rows.columns:
            return
        for cfg in configs:
            sub = rows[rows["config"] == cfg].dropna(subset=[metric_name]).sort_values("test_year")
            if sub.empty:
                continue
            years = sub["test_year"].to_numpy()
            values = sub[metric_name].to_numpy(dtype=float)
            label = cfg if with_label else None
            lo_col, hi_col = f"{metric_name}_CI_lower", f"{metric_name}_CI_upper"
            if with_ci and lo_col in sub.columns and hi_col in sub.columns:
                lo = np.where(sub[lo_col].isna(), values, sub[lo_col].to_numpy(dtype=float))
                hi = np.where(sub[hi_col].isna(), values, sub[hi_col].to_numpy(dtype=float))
                yerr = np.clip(np.vstack([values - lo, hi - values]), 0, None)
                ax.errorbar(years, values, yerr=yerr, fmt=f"{marker}{linestyle}", capsize=3,
                            color=colors[cfg], alpha=0.9, label=label)
            else:
                ax.plot(years, values, f"{marker}{linestyle}", color=colors[cfg], alpha=0.8, label=label)
        for hy in holdout_years:
            ax.axvline(hy, color="black", linestyle=":", alpha=0.35)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 9), sharex=True)

    # Each panel overlays two metrics (solid + dashed line, same color per
    # config). A second explicit legend (black Line2D proxies) documents which
    # line style/marker is each metric, so the two legends together cover
    # EVERYTHING plotted: color -> config, line style/marker -> metric.
    _plot_series(axes[0], "NSE", "o", "-", with_ci=True, with_label=True)
    _plot_series(axes[0], "KGE", "s", "--", with_ci=True, with_label=False)
    axes[0].set_ylabel("NSE / KGE")
    axes[0].set_title(f"Year-over-year evolution: NSE/KGE — M*={best_config} vs M1 and baselines")

    _plot_series(axes[1], "HitRatio", "o", "-", with_ci=True, with_label=True)
    _plot_series(axes[1], "FARate", "s", "--", with_ci=False, with_label=False)
    axes[1].set_ylabel("HitRatio / FARate")
    axes[1].set_title("Year-over-year evolution: HitRatio/FARate")
    axes[1].set_xlabel("Test year (vertical dotted line = holdout)")

    # Config legend (color) taken from axes[0]: identical in both panels, so it
    # is shown once below the figure instead of overlapping the data twice.
    config_handles, config_labels = axes[0].get_legend_handles_labels()

    metric_handles = {
        axes[0]: [Line2D([0], [0], color="black", marker="o", linestyle="-", label="NSE"),
                  Line2D([0], [0], color="black", marker="s", linestyle="--", label="KGE")],
        axes[1]: [Line2D([0], [0], color="black", marker="o", linestyle="-", label="HitRatio"),
                  Line2D([0], [0], color="black", marker="s", linestyle="--", label="FARate")],
    }
    for ax in axes:
        ax.legend(handles=metric_handles[ax], loc="upper right", frameon=False, title="Metric", fontsize=9)

    fig.tight_layout()
    fig.legend(config_handles, config_labels, loc="lower center", ncol=min(len(configs), 5),
               frameon=False, title="Config", bbox_to_anchor=(0.5, -0.03))
    _save_fig(fig, "Figure_MultiYear", doc_dir)
    return doc_dir / "Figure_MultiYear"

# --------------------------------------------------------------------------- #
# Figure_Ablation: M1-M5 vs. baselines
# --------------------------------------------------------------------------- #

def figure_ablation(df: pd.DataFrame, metric: str = "NSE", doc_dir: Path = DOC_DIR) -> Optional[Path]:
    """Bar chart of M1-M5 and baselines (mean ± 95% CI across expanding-CV folds).

    Restricted to ``strategy="expanding"`` so that folds of strategies with
    different year semantics (expanding 2017-2023 vs. LOYO 2014-2023) are
    never mixed in the same mean.

    Args:
        df (pd.DataFrame): Published-variant results.
        metric (str): Metric to plot.
        doc_dir (Path): Output directory.

    Returns:
        Optional[Path]: Base path of the saved figure, or ``None``.
    """
    if metric not in df.columns:
        print(f"[Figure_Ablation] Metric '{metric}' missing from results_consolidated.csv; skipped.")
        return None
    df = _filter_horizon(df)
    order = ABLATION_ORDER + BASELINE_ORDER
    base = df[df["strategy"] == "expanding"] if "strategy" in df.columns else df
    sub = base[base["config"].isin(order) & base[metric].notna()].copy()
    if sub.empty:
        print("[Figure_Ablation] No ablation/baseline rows (or metric missing); skipped.")
        return None

    present = [c for c in order if c in sub["config"].unique()]
    agg = sub.groupby("config")[metric].agg(["mean", "std", "count"]).reindex(present)
    agg["sem"] = (agg["std"] / np.sqrt(agg["count"].clip(lower=1))).fillna(0.0)
    agg["ci95"] = 1.96 * agg["sem"]

    colors = ["#4c72b0" if c in ABLATION_ORDER else "#55a868" for c in present]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(present, agg["mean"].to_numpy(), yerr=agg["ci95"].to_numpy(), capsize=4, color=colors)
    ax.set_ylabel(f"{metric} (mean ± 95% CI across folds)")
    ax.set_title(f"Ablation M1-M5 and baselines ({metric})")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=25, ha="right")
    fig.tight_layout()

    _save_fig(fig, "Figure_Ablation", doc_dir)
    return doc_dir / "Figure_Ablation"


# --------------------------------------------------------------------------- #
# Figure_Ablation_Radar: M1-M5 radar chart, 5 normalized axes
# --------------------------------------------------------------------------- #

def figure_ablation_radar(df: pd.DataFrame, doc_dir: Path = DOC_DIR) -> Optional[Path]:
    """M1-M5 radar chart on five axes normalized to [0, 1] over the observed range.

    Axes: KGE, HitRatio, F1, 1-FARate and 1/|PeakTiming_mean_lag|. A mean lag
    of exactly 0 (perfect timing) is assigned the maximum 1/|lag| of the other
    configurations instead of dividing by zero.

    Args:
        df (pd.DataFrame): Published-variant results.
        doc_dir (Path): Output directory.

    Returns:
        Optional[Path]: Base path of the saved figure, or ``None``.
    """
    metrics_needed = ["KGE", "HitRatio", "F1", "FARate", "PeakTiming_mean_lag"]
    df = _filter_horizon(df)
    sub = df[df["config"].isin(ABLATION_ORDER)]
    present = [c for c in metrics_needed if c in sub.columns]
    if len(present) < len(metrics_needed):
        print(f"[Figure_Ablation_Radar] Missing metrics {set(metrics_needed) - set(present)}; skipped.")
        return None

    means = sub.groupby("config")[present].mean(numeric_only=True).reindex(ABLATION_ORDER).dropna(how="all")
    if means.empty:
        print("[Figure_Ablation_Radar] No M1-M5 rows; skipped.")
        return None

    inv_lag = 1.0 / means["PeakTiming_mean_lag"].abs()
    finite_inv_lag = inv_lag.replace([np.inf, -np.inf], np.nan)
    cap = finite_inv_lag.max()
    inv_lag = inv_lag.replace([np.inf, -np.inf], cap)

    axes_data = pd.DataFrame({
        "KGE": means["KGE"],
        "HitRatio": means["HitRatio"],
        "F1": means["F1"],
        "1-FARate": 1.0 - means["FARate"],
        "1/|PeakTiming|": inv_lag,
    })
    axes_range = (axes_data.max() - axes_data.min()).replace(0, 1)
    axes_norm = ((axes_data - axes_data.min()) / axes_range).fillna(0.0)

    labels = list(axes_data.columns)
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection="polar"))
    colors = sns.color_palette("tab10", n_colors=len(axes_norm.index))
    for cfg, color in zip(axes_norm.index, colors):
        values = axes_norm.loc[cfg].tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=1.8, label=cfg)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("M1-M5 ablation radar (normalized axes [0,1])", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False)
    fig.tight_layout()

    _save_fig(fig, "Figure_Ablation_Radar", doc_dir)
    return doc_dir / "Figure_Ablation_Radar"


# --------------------------------------------------------------------------- #
# Figure_Robustness: dispersion under ±5%/±10% rainfall noise
# --------------------------------------------------------------------------- #

def figure_robustness(predictions_dir: Path = PREDICTIONS_DIR, doc_dir: Path = DOC_DIR) -> Optional[Path]:
    """Dispersion of the individual Monte Carlo replicas per noise type.

    Each point is the deviation of one replica from the unperturbed baseline
    prediction (drawn as the reference at 0), rather than an already collapsed
    summary.

    Args:
        predictions_dir (Path): Directory with ``robustness_*.npz``.
        doc_dir (Path): Output directory.

    Returns:
        Optional[Path]: Base path of the saved figure, or ``None``.
    """
    files = sorted(predictions_dir.glob("robustness_*.npz")) if predictions_dir.exists() else []
    if not files:
        print("[Figure_Robustness] No robustness_*.npz files; skipped.")
        return None

    frames = []
    for f in files:
        # pattern: robustness_{config}_{uniform|gaussian}_{year}.npz
        parts = f.stem.split("_")
        if len(parts) != 4:
            print(f"[Figure_Robustness] {f.name}: unexpected file name, skipped.")
            continue
        _prefix, config, noise_type, year = parts
        with np.load(f) as data:
            predictions = data["predictions"]  # (n_replicas, N)
            baseline = data["baseline"]  # (N,)
            deviation = (predictions - baseline[None, :]).ravel()
            frames.append(pd.DataFrame({
                "config": config,
                "noise_type": "Uniform ±10%" if noise_type == "uniform" else "Gaussian ±5%",
                "year": year,
                "deviation": deviation,
            }))
    if not frames:
        print("[Figure_Robustness] No file with a recognizable name; skipped.")
        return None
    long_df = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.boxplot(data=long_df, x="year", y="deviation", hue="noise_type", ax=ax)
    ax.axhline(0.0, color="black", linewidth=1.2, linestyle="--", alpha=0.7, label="Baseline (unperturbed)")
    ax.set_ylabel("Qe deviation from baseline (m³/s)")
    ax.set_xlabel("Year")
    configs = sorted(long_df["config"].unique())
    ax.set_title(f"Dispersion of the 50 MC replicas under rainfall perturbation ({', '.join(configs)})")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title=None, frameon=False)
    fig.tight_layout()

    _save_fig(fig, "Figure_Robustness", doc_dir)
    return doc_dir / "Figure_Robustness"


# --------------------------------------------------------------------------- #
# Figure_Robustness_Envelope / Figure_Robustness_Magnitude: complementary views
# to Figure_Robustness (with a single holdout year, the per-year box is squashed
# near 0 and dominated by peak-event outliers). Both read saved artifacts only
# (robustness_*.npz + predictions_cache/pred_*.npz, joined by hash through
# results_consolidated.csv) -- no retraining, no metric recomputation.
# --------------------------------------------------------------------------- #

def _robustness_file_groups(predictions_dir: Path) -> dict[tuple[str, str], dict[str, Path]]:
    """Group the robustness files by (config, year).

    Args:
        predictions_dir (Path): Directory with
            ``robustness_{config}_{uniform|gaussian}_{year}.npz``.

    Returns:
        dict[tuple[str, str], dict[str, Path]]: ``{(config, year): {noise_type: path}}``.
    """
    groups: dict[tuple[str, str], dict[str, Path]] = {}
    files = sorted(predictions_dir.glob("robustness_*.npz")) if predictions_dir.exists() else []
    for f in files:
        parts = f.stem.split("_")
        if len(parts) != 4:
            continue
        _prefix, config, noise_type, year = parts
        groups.setdefault((config, year), {})[noise_type] = f
    return groups


def _holdout_dates_and_observed(df: pd.DataFrame, config: str, year: str, horizon: str = "48h"):
    """Dates and observed inflow of the fold that produced the robustness baseline.

    Joins results_consolidated.csv (config, test_year) -> hash ->
    ``predictions_cache/pred_{algo_tag}.npz``, the same cache used by
    generate_yearly_plots.py. The model tag uses the first TRAINED penalty
    (penalty 0 is never trained for ``pinball_from_penalty``), and rows are
    restricted to one horizon so that ``rows.iloc[0]`` is unambiguous.

    Args:
        df (pd.DataFrame): Published-variant results.
        config (str): Configuration.
        year (str): Test year.
        horizon (str): Horizon to keep.

    Returns:
        ``(dates, y_true)``, or ``None`` if no fold or cache is available
        (nothing is ever retrained to fill it).
    """
    rows = df[(df["config"] == config) & (df["test_year"] == int(year))]
    if "strategy" in rows.columns and (rows["strategy"] == "holdout_final").any():
        rows = rows[rows["strategy"] == "holdout_final"]
    if "horizon" in rows.columns:
        rows = rows[rows["horizon"] == horizon]
    if rows.empty:
        return None
    cfg_meta = next((c for c in ABLATION_CONFIGS if c["name"] == config), None)
    if cfg_meta is None:
        return None
    trained_penalties = [p for p in cfg_meta["penalty"] if p != 0] or cfg_meta["penalty"]
    algo_tag = f"{cfg_meta['loss_name']}_{int(trained_penalties[0])}"
    cache_dir = Path(CONFIG.paths["cache_root"]) / str(rows.iloc[0]["hash"]) / "predictions_cache"
    npz_path = cache_dir / f"pred_{algo_tag}.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=True) as z:
        return pd.to_datetime(z["dates"]), z["y_true"].astype(float)


def figure_robustness_envelope(df: pd.DataFrame, predictions_dir: Path = PREDICTIONS_DIR,
                                doc_dir: Path = DOC_DIR) -> List[Path]:
    """[p5, p95] envelope of the Monte Carlo replicas over the holdout time series.

    Plotted with the observed inflow and the unperturbed baseline, it shows
    WHERE in the hydrograph the rainfall perturbation matters most (expected:
    flood peaks).

    Args:
        df (pd.DataFrame): Published-variant results.
        predictions_dir (Path): Directory with ``robustness_*.npz``.
        doc_dir (Path): Output directory.

    Returns:
        List[Path]: Base paths of the saved figures.
    """
    saved: List[Path] = []
    noise_label = {"uniform": "Uniform ±10%", "gaussian": "Gaussian ±5%"}
    for (config, year), files in _robustness_file_groups(predictions_dir).items():
        ref = _holdout_dates_and_observed(df, config, year)
        if ref is None:
            print(f"[Figure_Robustness_Envelope] {config}_{year}: no dates/y_true in predictions_cache; skipped.")
            continue
        dates, y_true = ref

        noise_order = [n for n in ("uniform", "gaussian") if n in files]
        if not noise_order:
            continue
        fig, axes = plt.subplots(nrows=len(noise_order), ncols=1,
                                  figsize=(12, 4.2 * len(noise_order)), sharex=True)
        axes = np.atleast_1d(axes)

        plotted = 0
        for ax, noise_type in zip(axes, noise_order):
            with np.load(files[noise_type]) as z:
                predictions = z["predictions"].astype(float)
                baseline = z["baseline"].astype(float)
            if predictions.shape[1] != len(dates):
                print(f"[Figure_Robustness_Envelope] {config}_{year}_{noise_type}: lengths are not aligned; skipped.")
                continue
            p5, p95 = np.percentile(predictions, [5, 95], axis=0)

            ax.plot(dates, y_true, color="black", linewidth=1.2, alpha=0.85, label="Observed")
            ax.plot(dates, baseline, color="tab:blue", linewidth=1.3, linestyle="--", label="Baseline (unperturbed)")
            ax.fill_between(dates, p5, p95, color="tab:orange", alpha=0.3, label="MC replicas [p5, p95]")
            ax.set_ylabel("Inflow (m³/s)")
            ax.set_title(f"{config} — {noise_label.get(noise_type, noise_type)}", fontsize=11)
            ax.legend(frameon=False, loc="upper right", fontsize=9)
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            continue

        axes[-1].set_xlabel("Date")
        fig.suptitle(f"Rainfall-perturbation envelope over the test period — {config} ({year})")
        fig.tight_layout()

        name = f"Figure_Robustness_Envelope_{config}_{year}"
        _save_fig(fig, name, doc_dir)
        saved.append(doc_dir / name)

    if not saved:
        print("[Figure_Robustness_Envelope] No folds with dates/y_true available; skipped.")
    return saved


def figure_robustness_magnitude(predictions_dir: Path = PREDICTIONS_DIR,
                                 doc_dir: Path = DOC_DIR) -> Optional[Path]:
    """Replica deviation vs. the MAGNITUDE of the baseline inflow at each time step.

    Shows whether the effect of the rainfall perturbation scales with event
    intensity (expected heteroscedasticity: little noise at low flow, much at
    peaks).

    Args:
        predictions_dir (Path): Directory with ``robustness_*.npz``.
        doc_dir (Path): Output directory.

    Returns:
        Optional[Path]: Base path of the saved figure, or ``None``.
    """
    frames = []
    for (config, _year), files in _robustness_file_groups(predictions_dir).items():
        for noise_type, path in files.items():
            with np.load(path) as z:
                predictions = z["predictions"].astype(float)
                baseline = z["baseline"].astype(float)
            deviation = predictions - baseline[None, :]
            frames.append(pd.DataFrame({
                "baseline": np.tile(baseline, predictions.shape[0]),
                "deviation": deviation.ravel(),
                "noise_type": "Uniform ±10%" if noise_type == "uniform" else "Gaussian ±5%",
                "config": config,
            }))
    if not frames:
        print("[Figure_Robustness_Magnitude] No robustness_*.npz files; skipped.")
        return None
    long_df = pd.concat(frames, ignore_index=True)

    noise_types = sorted(long_df["noise_type"].unique())
    fig, axes = plt.subplots(nrows=1, ncols=len(noise_types),
                              figsize=(6.5 * len(noise_types), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, noise_type in zip(axes, noise_types):
        sub = long_df[long_df["noise_type"] == noise_type]
        hb = ax.hexbin(sub["baseline"], sub["deviation"], gridsize=40, cmap="viridis", mincnt=1, bins="log")
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_xlabel("Baseline inflow (m³/s)")
        ax.set_title(noise_type, fontsize=11)
        fig.colorbar(hb, ax=ax, label="log10(count)")
    axes[0].set_ylabel("Replica deviation from baseline (m³/s)")

    configs = sorted(long_df["config"].unique())
    fig.suptitle(f"Perturbation deviation vs. baseline inflow magnitude ({', '.join(configs)})")
    fig.tight_layout()

    _save_fig(fig, "Figure_Robustness_Magnitude", doc_dir)
    return doc_dir / "Figure_Robustness_Magnitude"


# --------------------------------------------------------------------------- #
# Figure_Arch_Unified: Encoder-Decoder + masking pretraining + RF selector diagram
# --------------------------------------------------------------------------- #

def figure_arch_unified(doc_dir: Path = DOC_DIR) -> Path:
    """Architecture diagram.

    Input block (4 rain gauges + 24h/48h/72h forecasts), encoder and decoder,
    and the three phases (pretraining, multi-branch asymmetric fine-tuning and
    dynamic RF selection).

    Args:
        doc_dir (Path): Output directory.

    Returns:
        Path: Base path of the saved figure.
    """
    fig, ax = plt.subplots(figsize=(14.5, 4.2))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.2, 1.0, 1.9, 2.0, "Input\n4 rain gauge stations\n+ 24h/48h/72h forecast", "#e6e6e6"),
        (2.5, 1.0, 2.2, 2.0, "Encoder\n(LSTM, historical context)", "#cfe8ff"),
        (5.2, 1.0, 2.2, 2.0, "Decoder\n(LSTM, forecast horizon)", "#cfe8ff"),
        (8.0, 2.35, 2.9, 1.25, "Phase 1: Pretrain\n(Qe=0 masking)", "#ffe8b3"),
        (8.0, 0.4, 2.9, 1.25, "Phase 2: Finetune\n(multi-branch asymmetric loss)", "#ffe8b3"),
        (11.4, 1.0, 2.9, 2.0, "Phase 3: RF Selector\n(dynamic selection,\nordinal regression)", "#c8f5c8"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    arrow_kwargs = dict(arrowstyle="-|>", lw=1.6, color="black", mutation_scale=16)
    connections = [
        ((2.1, 2.0), (2.5, 2.0)),
        ((4.7, 2.0), (5.2, 2.0)),
        ((7.4, 2.0), (8.0, 2.975)),
        ((7.4, 2.0), (8.0, 1.025)),
        ((10.9, 2.975), (11.4, 2.0)),
        ((10.9, 1.025), (11.4, 2.0)),
    ]
    for (x0, y0), (x1, y1) in connections:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=arrow_kwargs)

    ax.set_title("Unified architecture: Input + Encoder-Decoder + Masking Pretrain + RF Selector", fontsize=12)
    fig.tight_layout()

    _save_fig(fig, "Figure_Arch_Unified", doc_dir)
    return doc_dir / "Figure_Arch_Unified"


# --------------------------------------------------------------------------- #
# LaTeX tables (booktabs)
# --------------------------------------------------------------------------- #

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "_": r"\_",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
    "±": r"$\pm$",
}


def _escape_latex(s: str) -> str:
    """Escape LaTeX special characters (and ``±``) in a string."""
    out = []
    for ch in s:
        out.append(_LATEX_ESCAPES.get(ch, ch))
    return "".join(out)


def _df_to_booktabs_latex(df: pd.DataFrame, caption: str, label: str, index: bool = True) -> str:
    """Render a DataFrame as a booktabs LaTeX table, by hand.

    Avoids ``DataFrame.to_latex``/``Styler``, which require a newer jinja2 than
    the one installed.

    Args:
        df (pd.DataFrame): Table content.
        caption (str): Caption (escaped).
        label (str): LaTeX label.
        index (bool): Include the index as the first column.

    Returns:
        str: LaTeX source.
    """
    header_cells = ([_escape_latex(df.index.name or "")] if index else []) + [
        _escape_latex(str(c)) for c in df.columns
    ]
    col_format = ("l" if index else "") + "r" * len(df.columns)

    def _fmt(v) -> str:
        if isinstance(v, (float, np.floating)):
            return "--" if np.isnan(v) else f"{v:.3f}"
        return _escape_latex(str(v))

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{_escape_latex(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_format}}}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]
    for idx, row in df.iterrows():
        cells = ([_escape_latex(str(idx))] if index else []) + [_fmt(v) for v in row.tolist()]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# Full metric columns, with the bootstrap 95% confidence interval (computed by
# log_extended_fold_metrics) for the metrics in _CI_METRICS.
_TABLE_METRICS = ["NSE", "KGE", "HitRatio", "FARate", "FARatio", "F1", "Precision", "Recall",
                   "PeakTiming_mean_lag", "PeakTiming_mean_absolute_lag", "PeakTiming_n_peaks"]
_CI_METRICS = ("NSE", "KGE", "HitRatio", "F1")


def _build_metrics_table(df: pd.DataFrame, configs: List[str]) -> pd.DataFrame:
    """Metrics table shared by the CV table and the holdout table.

    One row per configuration (in the given order); NSE/KGE/HitRatio/F1 are
    formatted as ``mean [95% CI lo, hi]`` from the bootstrap columns, the rest
    of ``_TABLE_METRICS`` as plain values.

    Args:
        df (pd.DataFrame): Rows already filtered (expanding-CV year range, or
            ``strategy="holdout_final"``).
        configs (List[str]): Configurations, in order.

    Returns:
        pd.DataFrame: The table (empty if no configuration has rows).
    """
    present = [c for c in _TABLE_METRICS if c in df.columns]
    rows = {}
    for cfg in configs:
        sub = df[df["config"] == cfg]
        if sub.empty:
            continue
        row = {}
        for m in present:
            val = sub[m].mean()
            lo_col, hi_col = f"{m}_CI_lower", f"{m}_CI_upper"
            if m in _CI_METRICS and lo_col in sub.columns and hi_col in sub.columns:
                lo, hi = sub[lo_col].mean(), sub[hi_col].mean()
                if pd.notna(val) and pd.notna(lo) and pd.notna(hi):
                    row[m] = f"{val:.3f} [{lo:.3f}, {hi:.3f}]"
                elif pd.notna(val):
                    row[m] = f"{val:.3f}"
                else:
                    row[m] = "--"
            else:
                row[m] = round(float(val), 3) if pd.notna(val) else float("nan")
        rows[cfg] = row
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).T.reindex([c for c in configs if c in rows])
    table.index.name = "Config"
    return table


def _table_configs(best_config: str) -> List[str]:
    """M*, the baselines and M1, in that order, without duplicates if M* is M1."""
    order = [best_config] + BASELINE_ORDER + ["M1"]
    seen = set()
    return [c for c in order if not (c in seen or seen.add(c))]


def _dual_regime_configs(best_config: str) -> List[str]:
    """Configurations of the dual-regime tables.

    M5 (emergency regime, alpha <= 0.12) followed by ``_table_configs``
    (standard regime, alpha <= 0.10: M*, baselines, M1). Instead of raising the
    FARate threshold until M5 wins (threshold shopping), both regimes are
    reported side by side with their real metrics, including M5's low NSE.
    """
    order = ["M5"] + _table_configs(best_config)
    seen = set()
    return [c for c in order if not (c in seen or seen.add(c))]


def _build_ablation_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """M1-M5 ablation matrix: design metadata joined with mean metrics.

    Design metadata (pretraining, loss, RF) is read live from
    ``ABLATION_CONFIGS`` rather than duplicated as CSV columns that could go
    out of sync. Metrics are means over all available folds (expanding and
    LOYO, no year filter): a design snapshot, not a temporal comparison.

    Args:
        df (pd.DataFrame): Results for one horizon.

    Returns:
        Optional[pd.DataFrame]: The table, or ``None``.
    """
    metrics_cols = ["NSE", "KGE", "HitRatio", "FARate", "F1", "PeakTiming_mean_lag"]
    present = [c for c in metrics_cols if c in df.columns]
    means = df[df["config"].isin(ABLATION_ORDER)].groupby("config")[present].mean(numeric_only=True).round(3)

    rows = {}
    for cfg in ABLATION_ORDER:
        meta = next((c for c in ABLATION_CONFIGS if c["name"] == cfg), None)
        if meta is None:
            continue
        # penalty=0 is dropped for pinball_from_penalty in
        # Iteration._expand_valid_grid and never trained, so it must not appear
        # as part of the range actually used.
        trained_penalties = [p for p in meta["penalty"] if p != 0] or meta["penalty"]
        if meta["loss_name"] == "original_mae":
            loss_desc = "MAE"
        elif len(trained_penalties) > 1:
            loss_desc = f"Pinball (p={trained_penalties[0]}-{trained_penalties[-1]})"
        else:
            loss_desc = f"Pinball (p={trained_penalties[0]})"
        row = {
            "Pretraining (Qe=0)": "Yes" if meta["use_pretrain"] else "No",
            "Loss": loss_desc,
            "RF": "Yes" if meta["use_rf"] else "No",
        }
        if cfg in means.index:
            row.update(means.loc[cfg].to_dict())
        rows[cfg] = row
    if not rows:
        return None
    table = pd.DataFrame(rows).T.reindex(ABLATION_ORDER)
    table.index.name = "Config"
    return table


_DEGRADATION_METRICS = ["NSE", "KGE", "HitRatio", "FARate", "F1", "PeakTiming_mean_lag"]
_HORIZON_ORDER = ["24h", "48h", "72h"]


def _build_horizon_degradation_table(df: pd.DataFrame, configs: List[str], strategy: str = "expanding",
                                      year_range: Optional[tuple] = None) -> Optional[pd.DataFrame]:
    """Generic (config, horizon) -> metrics table across forecast lead times.

    Used for the expanding-CV lead-time degradation table and the multi-horizon
    blind-holdout table. Only horizons actually present in the CSV get rows.
    Nothing is retrained or recomputed; rows are only grouped.

    Args:
        df (pd.DataFrame): Results for all horizons.
        configs (List[str]): Configurations, in order.
        strategy (str): Strategy to keep.
        year_range (Optional[tuple]): Inclusive ``test_year`` range, if any.

    Returns:
        Optional[pd.DataFrame]: The table, or ``None``.
    """
    if "horizon" not in df.columns:
        return None
    base = df[df["strategy"] == strategy] if "strategy" in df.columns else df
    if year_range is not None:
        base = base[base["test_year"].between(*year_range)]

    present = [c for c in _DEGRADATION_METRICS if c in base.columns]
    rows = []
    for horizon in _HORIZON_ORDER:
        sub_h = base[base["horizon"] == horizon]
        if sub_h.empty:
            continue
        for cfg in configs:
            sub = sub_h[sub_h["config"] == cfg]
            if sub.empty:
                continue
            row = {"Config": cfg, "Lead Time": horizon}
            for m in present:
                val = sub[m].mean()
                row[m] = round(float(val), 3) if pd.notna(val) else float("nan")
            rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("Config")


def export_latex_tables(df: Optional[pd.DataFrame], doc_dir: Path = DOC_DIR) -> Path:
    """Write every manuscript table to ``<doc_dir>/tables_latex.tex``.

    Args:
        df (Optional[pd.DataFrame]): Published-variant results (all horizons).
        doc_dir (Path): Output directory.

    Returns:
        Path: Path of the ``.tex`` file.
    """
    doc_dir.mkdir(parents=True, exist_ok=True)
    out_path = doc_dir / "tables_latex.tex"
    metrics_cols = ["NSE", "KGE", "HitRatio", "FARate", "FARatio", "F1", "PeakTiming_mean_absolute_lag"]

    parts = ["% Auto-generated by generate_paper_figures.py -- PHASE 3, Step 2\n"]

    if df is None or df.empty:
        parts.append("% results_consolidated.csv not available or empty when the tables were generated.\n")
        out_path.write_text("\n".join(parts), encoding="utf-8")
        print(f"[export_latex_tables] {out_path} (no data)")
        return out_path

    present_cols = [c for c in metrics_cols if c in df.columns]
    best_config = _best_config()
    # Tables 1-4 are restricted to 48h (the validated horizon); only the
    # lead-time tables use the full multi-horizon `df`.
    df_48h = _filter_horizon(df)

    # --- Table 1: expanding-window CV 2017-2023 (dual regime: M5 + M*/baselines/M1) ---
    if "strategy" in df_48h.columns:
        cv = df_48h[(df_48h["strategy"] == "expanding")
                    & df_48h["test_year"].between(*EXPANDING_CV_YEAR_RANGE)]
    else:
        cv = df_48h[df_48h["test_year"].between(*EXPANDING_CV_YEAR_RANGE)]
    # Dual regime: M5 is added to M*/baselines/M1, not as a replacement for M*
    # (see _dual_regime_configs).
    table1 = _build_metrics_table(cv, _dual_regime_configs(best_config))
    if not table1.empty:
        parts.append(f"% Table 1: Expanding-window CV (test_year {EXPANDING_CV_YEAR_RANGE[0]}-"
                      f"{EXPANDING_CV_YEAR_RANGE[1]}), M*={best_config}\n")
        parts.append(_df_to_booktabs_latex(
            table1, caption=f"Expanding-window CV (test_year {EXPANDING_CV_YEAR_RANGE[0]}-"
                             f"{EXPANDING_CV_YEAR_RANGE[1]}): mean [95% CI] across folds.",
            label="tab:cv_expanding",
        ))
        parts.append(
            "\n\\noindent\\textit{Note: two operating regimes are reported side by side, not a "
            "single selected model. Standard regime (alpha <= 0.10 on FARate): selects "
            f"{best_config} by maximizing DeltaHitRatio subject to NSE > 0 and FARate <= 0.10. "
            "Emergency early-warning regime (alpha <= 0.12): admits M5 (the full RF-selector "
            "framework), which is excluded under the standard threshold (FARate = 0.103) but "
            "reaches a substantially higher HitRatio at the cost of a much lower NSE -- see M5's "
            "row above for the exact trade-off, not an idealized one.}\\par\n"
        )

    # --- Table 1b: summary per configuration; Table 1c: M* by year ---
    summary = df_48h.groupby("config")[present_cols].mean(numeric_only=True).round(3)
    order = [c for c in (ABLATION_ORDER + BASELINE_ORDER) if c in summary.index]
    # The tau=0.90 baseline variants are excluded from the summary table (which
    # uses the canonical tau=0.50 baselines); they are reported in the
    # dedicated tau=0.50 vs tau=0.90 comparison.
    extra = [c for c in summary.index if c not in order and not c.endswith("_tau90")]
    summary = summary.reindex(order + extra)
    summary.index.name = "Config"
    parts.append("\n% Table 1b: metrics summary per configuration (mean across all folds)\n")
    parts.append(_df_to_booktabs_latex(
        summary, caption="Metrics summary per configuration (mean across all folds).",
        label="tab:summary_metrics",
    ))

    if best_config in df_48h["config"].unique():
        best_rows = df_48h[df_48h["config"] == best_config].copy()
        # The same test_year can appear under several strategies (expanding and
        # LOYO overlap), so "Year" includes the strategy to identify rows uniquely.
        if "strategy" in best_rows.columns:
            best_rows["Year"] = best_rows["test_year"].astype(str) + " (" + best_rows["strategy"] + ")"
        else:
            best_rows["Year"] = best_rows["test_year"].astype(str)
        best_rows = best_rows.sort_values(["test_year", "strategy"] if "strategy" in best_rows.columns
                                           else ["test_year"])
        per_year_cols = ["Year"] + present_cols
        per_year_cols = [c for c in per_year_cols if c in best_rows.columns]
        per_year = best_rows[per_year_cols].round(3).set_index("Year")
        parts.append(f"\n% Table 1c: {best_config} by year\n")
        parts.append(_df_to_booktabs_latex(
            per_year, caption=f"{best_config}: metrics by year and strategy.",
            label="tab:best_config_per_year",
        ))

    # --- Table 2: M1-M5 ablation matrix ---
    table2 = _build_ablation_table(df_48h)
    if table2 is not None:
        parts.append("\n% Table 2: M1-M5 ablation matrix\n")
        parts.append(_df_to_booktabs_latex(
            table2, caption="M1-M5 ablation matrix: design configuration and mean metrics.",
            label="tab:ablation_matrix",
        ))

    # --- Table 3: blind holdout, dual regime ---
    if "strategy" in df_48h.columns:
        holdout = df_48h[df_48h["strategy"] == "holdout_final"]
        table3 = _build_metrics_table(holdout, _dual_regime_configs(best_config))
        if not table3.empty:
            n_extreme = holdout.loc[holdout["config"] == best_config, "n_extreme"]
            n_extreme_note = f" ({int(n_extreme.iloc[0])} real extreme events evaluated)" \
                if not n_extreme.empty and pd.notna(n_extreme.iloc[0]) else ""
            parts.append(f"\n% Table 3: blind holdout (D_holdout), M*={best_config}{n_extreme_note}\n")
            parts.append(_df_to_booktabs_latex(
                table3, caption=f"Blind holdout (D_holdout): a single evaluation point per "
                                 f"config{n_extreme_note}.",
                label="tab:holdout",
            ))

    # --- Table 3b: blind holdout across the three lead times, dual regime ---
    table3b = _build_horizon_degradation_table(df, _dual_regime_configs(best_config), strategy="holdout_final")
    if table3b is not None:
        parts.append(f"\n% Table 3b: blind holdout across lead times (24h/48h/72h), dual regime\n")
        parts.append(_df_to_booktabs_latex(
            table3b, caption="Blind holdout (D_holdout) across forecast lead times "
                              "(24h/48h/72h): standard regime "
                              f"({best_config}, alpha <= 0.10) vs. emergency early-warning regime "
                              "(M5, alpha <= 0.12), against the same baselines and M1.",
            label="tab:holdout_horizon",
        ))

    # --- Table 4: Monte Carlo robustness ---
    robustness_table = _build_robustness_table(df_48h)
    if robustness_table is not None:
        table4, robustness_configs = robustness_table
        configs_label = " vs. ".join(robustness_configs)
        parts.append(f"\n% Table 4: Monte Carlo robustness ({configs_label})\n")
        parts.append(_df_to_booktabs_latex(
            table4, caption=f"Robustness under Monte Carlo rainfall perturbation: "
                             f"{configs_label} (N=50 replicas).",
            label="tab:robustness",
        ))

    # --- Table 5: lead-time degradation 24h/48h/72h on the expanding CV (M4 is
    # added to the dual-regime configurations to keep the M4 vs M5 vs M*
    # comparison). ---
    degradation_configs = ["M4"] + _dual_regime_configs(best_config)
    _seen = set()
    degradation_configs = [c for c in degradation_configs if not (c in _seen or _seen.add(c))]
    table5 = _build_horizon_degradation_table(
        df, degradation_configs, strategy="expanding", year_range=EXPANDING_CV_YEAR_RANGE,
    )
    if table5 is not None:
        parts.append(f"\n% Table 5: lead-time degradation (M*={best_config})\n")
        parts.append(_df_to_booktabs_latex(
            table5, caption=f"Metric degradation across forecast lead times "
                             f"(24h/48h/72h), M*={best_config}.",
            label="tab:horizon_degradation",
        ))

    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[export_latex_tables] Saved {out_path}")
    return out_path


def _build_robustness_table(df: pd.DataFrame) -> Optional[tuple]:
    """Monte Carlo robustness table (dual regime).

    One row per (configuration, scenario) for EVERY robustness configuration
    in the CSV, M5 (emergency regime) first, read from the
    ``config="{config}_robustness_{type}"`` rows written by ``run_robustness``
    and ``run_robustness_m5``. "RF branch stability" is shown when available
    ("--" for configurations without the RF selector).

    Args:
        df (pd.DataFrame): Results for one horizon.

    Returns:
        Optional[tuple]: ``(table, configuration order)``, or ``None`` if there
        are no robustness rows yet.
    """
    robustness_rows = df[df["config"].str.contains("_robustness_", na=False)]
    if robustness_rows.empty:
        return None

    configs_present = sorted(robustness_rows["config"].str.replace(r"_robustness_.*$", "", regex=True).unique())
    order = [c for c in ("M5",) if c in configs_present] + [c for c in configs_present if c != "M5"]

    rows = []
    for cfg in order:
        for noise_type, label in (("uniform", "Uniform ±10%"), ("gaussian", "Gaussian ±5%")):
            sub = robustness_rows[robustness_rows["config"] == f"{cfg}_robustness_{noise_type}"]
            if sub.empty:
                continue
            r = sub.iloc[0]
            branch_stability = r.get("rf_branch_stability")
            rows.append({
                "Config": cfg, "Scenario": label, "N": int(r["n_replicas"]),
                "MAD": round(float(r["mean_abs_deviation_from_baseline"]), 3),
                "σ̄": round(float(r["mean_std_across_replicas"]), 3),
                "σ_max": round(float(r["max_std_across_replicas"]), 3),
                "ΔHitRatio": round(float(r["delta_hit_ratio"]), 3) if pd.notna(r.get("delta_hit_ratio")) else float("nan"),
                "ΔFARate": round(float(r["delta_farate"]), 3) if pd.notna(r.get("delta_farate")) else float("nan"),
                "RF branch stability": f"{round(float(branch_stability) * 100, 1)}%"
                    if pd.notna(branch_stability) else "--",
            })

    if not rows:
        return None

    table = pd.DataFrame(rows).set_index("Config")
    return table, order


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    """Generate every paper figure and the LaTeX tables."""
    parser = argparse.ArgumentParser(description="Paper figures and LaTeX tables")
    parser.add_argument("--metric", default="NSE", help="Main metric for Figure_MultiYear/Figure_Ablation")
    parser.add_argument("--results_csv", default=str(RESULTS_CSV))
    parser.add_argument("--predictions_dir", default=str(PREDICTIONS_DIR))
    parser.add_argument("--doc_dir", default=str(DOC_DIR))
    args = parser.parse_args()

    doc_dir = Path(args.doc_dir)
    df = load_results(Path(args.results_csv))

    generated = []
    if df is not None:
        for fn in (figure_multiyear, figure_ablation):
            r = fn(df, metric=args.metric, doc_dir=doc_dir)
            if r:
                generated.append(r)
        r = figure_ablation_radar(df, doc_dir=doc_dir)
        if r:
            generated.append(r)

    r = figure_robustness(Path(args.predictions_dir), doc_dir=doc_dir)
    if r:
        generated.append(r)
    if df is not None:
        generated.extend(figure_robustness_envelope(df, Path(args.predictions_dir), doc_dir=doc_dir))
    r = figure_robustness_magnitude(Path(args.predictions_dir), doc_dir=doc_dir)
    if r:
        generated.append(r)

    generated.append(figure_arch_unified(doc_dir=doc_dir))
    export_latex_tables(df, doc_dir=doc_dir)

    print(f"[generate_paper_figures] {len(generated)} figures generated in {doc_dir}")


if __name__ == "__main__":
    main()
