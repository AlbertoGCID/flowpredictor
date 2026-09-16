"""Legacy read-only evaluation report over the per-iteration caches (band plots and metric tables).

Not used to produce the published results (see run_experiments_phase3.py and
visualization/generate_paper_figures.py).
"""
from __future__ import annotations

import os
import re
import json
import glob
import math
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Project logger and configuration
from ...log.log_config import get_logger
from ..config import CONFIG

# Optional: hydroeval for NSE/KGE (not required for plots)
try:
    import hydroeval as he
except Exception:
    he = None

logger = get_logger()


# ----------------------------- Basic utilities -----------------------------

def _read_json(path: str) -> Optional[dict]:
    """Read a JSON file, returning ``None`` on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _canonical_base(name: str) -> str:
    """Strip a trailing ``_train`` or ``_test`` from a prediction base name."""
    if name.endswith("_train"):
        return name[:-6]
    if name.endswith("_test"):
        return name[:-5]
    return name

def _safe_np_load(path: str) -> Optional[Any]:
    """Load an ``.npz`` file, returning ``None`` on any error."""
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None

def _ensure_dir(p: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(p, exist_ok=True)


def plot_flow_rain_from_csv(
    csv_path: str = "flowpredictor/datasets/dataframe_original.csv",
    out_dir: str = "flowpredictor/results/figs_intro",
    base_name: str = "flow_rain_all_years",
    test_start: str = "2013-06-30",   # June 30, 2013
    test_end: str   = "2014-07-01",   # July 1, 2014 (inclusive)
    font_base: int = 16,
    font_title: int = 20,
    # --- NEW: control of y-limits ---
    match_scale_to: str = "inflow",   # "inflow" or "fixed"
    y_min: float | None = None,       # e.g., 1.0 if match_scale_to="fixed"
    y_max: float | None = None,       # e.g., 400.0 if match_scale_to="fixed"
    pad_pct: float = 0.05             # padding when match_scale_to="inflow"
) -> dict:
    """
    Paper figure:
      - Solid: Inflow (m³/s)
      - Dashed: Daily total rainfall (l/m²) = sum of 4 gauges (daily means per gauge → summed)
      - TEST segment highlighted (2013-06-30 to 2014-07-01)
      - Both left (inflow) and right (rainfall) axes share the SAME Y LIMITS (no data rescaling).
        * match_scale_to="inflow": both axes adopt inflow range (+ padding)
        * match_scale_to="fixed":  both axes adopt [y_min, y_max]
    Saves PNG, PDF, CSV (English).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Columns
    date_col = "Fecha"
    flow_col = "Qe"
    rain_cols = ["l/m2_arzua", "l/m2_olveda", "l/m2_serradofaro", "l/m2_melide"]

    # Dates
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        alt = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors="coerce")
        dates = dates.fillna(alt)
    if dates.isna().any():
        raise ValueError("Some dates in 'Fecha' could not be parsed.")

    # Daily aggregation (no normalization)
    tmp = df.copy()
    tmp["_day"] = dates.dt.normalize()
    daily = tmp.groupby("_day").agg(
        **({"Inflow (m³/s)": (flow_col, "mean")})
        | {c: (c, "mean") for c in rain_cols}
    ).reset_index().rename(columns={"_day": "Date"})
    daily["Average rainfall (l/m²)"] = daily[rain_cols].mean(axis=1)

    # TEST mask (inclusive)
    d = pd.to_datetime(daily["Date"])
    ts = pd.to_datetime(test_start)
    te = pd.to_datetime(test_end)
    is_test = (d >= ts) & (d <= te)

    # Styling
    plt.rcParams.update({
        "font.size": font_base,
        "axes.labelsize": font_base,
        "xtick.labelsize": font_base,
        "ytick.labelsize": font_base,
        "legend.fontsize": font_base,
        "axes.titlesize": font_title,
        "figure.dpi": 100
    })

    # Plot
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    c_q_train, c_q_test = "C0", "C2"
    c_p_train, c_p_test = "C1", "C3"

    # Inflow
    ax1.plot(d[~is_test], daily.loc[~is_test, "Inflow (m³/s)"],
             color=c_q_train, linewidth=1.8, label="Inflow (train)")
    ax1.plot(d[ is_test], daily.loc[ is_test, "Inflow (m³/s)"],
             color=c_q_test,  linewidth=2.2, label="Inflow (test)")

    # Rainfall
    ax2.plot(d[~is_test], daily.loc[~is_test, "Average rainfall (l/m²)"],
             color=c_p_train, linewidth=1.6, linestyle="--", label="Rainfall (train)")
    ax2.plot(d[ is_test], daily.loc[ is_test, "Average rainfall (l/m²)"],
             color=c_p_test,  linewidth=2.0, linestyle="--", label="Rainfall (test)")

    # Labels
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Inflow (m³/s)")
    ax2.set_ylabel("Daily average rainfall (l/m²)")
    ax1.set_title("Inflow and daily rainfall")
    ax1.grid(alpha=0.25, linewidth=0.7)

    # --- SAME Y-LIMITS on both axes (no rescaling of data) ---
    if match_scale_to == "fixed":
        if y_min is None or y_max is None:
            raise ValueError("Provide y_min and y_max when match_scale_to='fixed'.")
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
    else:
        # Use inflow dynamic range with padding
        inf = daily["Inflow (m³/s)"].to_numpy(dtype=float)
        lo = np.nanmin(inf)
        hi = np.nanmax(inf)
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = 0.0, 1.0
        span = hi - lo
        lo_pad = lo - pad_pct * span
        hi_pad = hi + pad_pct * span
        ax1.set_ylim(lo_pad, hi_pad)
        ax2.set_ylim(lo_pad, hi_pad)

    # Legend
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, frameon=False, loc="upper left")

    plt.tight_layout()
    png = os.path.join(out_dir, f"{base_name}.png")
    pdf = os.path.join(out_dir, f"{base_name}.pdf")
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()

    # Output CSV (English)
    out_csv = os.path.join(out_dir, f"{base_name}.csv")
    export = daily[["Date", "Inflow (m³/s)", "Average rainfall (l/m²)"]].copy()
    export["Is TEST (0/1)"] = is_test.astype(int).values
    export.to_csv(out_csv, index=False)

    return {"png": png, "pdf": pdf, "csv": out_csv, "days": int(len(export))}








# ----------------------------- Data structures ---------------------------

@dataclass
class IterationEval:
    """Standardized view of an iteration for evaluation, plots and tables."""
    hash_id: str
    iter_dir: str
    preds_dir: str
    models_dir: str
    ensemble_dir: str
    # Standardized (test) data:
    dates: Optional[pd.DatetimeIndex] = None
    y_true: Optional[np.ndarray] = None
    y_cls: Optional[np.ndarray] = None  # classifier prediction
    y_mae: Optional[np.ndarray] = None  # baseline (original_mae_0), if present
    band_min: Optional[np.ndarray] = None
    band_max: Optional[np.ndarray] = None
    # Meta
    algo: Optional[str] = None
    model_bases: Optional[List[str]] = None
    metrics_classifier: Optional[dict] = None  # classifier metrics JSON, if present


# ----------------------------- Evaluator class -------------------------------

class ExperimentEvaluator:
    """Collect iteration artifacts under ``tests{numero_prueba}/<hash>/`` and produce plots/tables.

    Read-only: it never generates missing model artifacts.

    Args:
        numero_prueba (str): Experiment number.
        result_root (Optional[str]): Results root (default: ``CONFIG.paths["result_path"]``).
    """
    def __init__(self, numero_prueba: str, result_root: Optional[str] = None):
        self.numero_prueba = str(numero_prueba)
        self.result_root = result_root or CONFIG.paths["result_path"]
        self.tests_dir = os.path.join(self.result_root, f"prueba{self.numero_prueba}",f"tests{self.numero_prueba}")
        self.out_dir = os.path.join(self.tests_dir, "_evaluation")
        _ensure_dir(self.out_dir)
        self.logger = logger

    # -------- Discovery and standardized loading --------

    def discover_iterations(self) -> List[str]:
        """Hash directories that contain a ``predictions_cache``."""
        if not os.path.isdir(self.tests_dir):
            self.logger.warning("[EVAL] tests dir does not exist: %s", self.tests_dir)
            return []
        cands = sorted([d for d in os.listdir(self.tests_dir)
                        if os.path.isdir(os.path.join(self.tests_dir, d)) and not d.startswith("_")])
        # Keep only directories with a predictions_cache
        hashes = []
        for h in cands:
            pc = os.path.join(self.tests_dir, h, "predictions_cache")
            if os.path.isdir(pc):
                hashes.append(h)
        if not hashes:
            self.logger.warning("[EVAL] No iterations with predictions_cache found in %s", self.tests_dir)
        return hashes

    def _load_classifier_meta(self, ensemble_dir: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """Meta-learner name and branch order from the first classifier metadata JSON."""
        cls_dir = os.path.join(ensemble_dir, "classifiers")
        if not os.path.isdir(cls_dir):
            return None, None
        # metadata: first *.json in classifiers/
        metas = glob.glob(os.path.join(cls_dir, "*.json"))
        if not metas:
            return None, None
        meta = _read_json(metas[0])
        if not meta:
            return None, None
        algo = meta.get("algo")
        bases = meta.get("model_bases") or meta.get("models") or []
        bases = [_canonical_base(b) for b in bases]
        return algo, bases

    def _collect_standard_data(self, h: str) -> Optional[IterationEval]:
        """Load the classifier and per-model NPZ/JSON files and build the min–max band."""
        iter_dir = os.path.join(self.tests_dir, h)
        preds_dir = os.path.join(iter_dir, "predictions_cache")
        models_dir = os.path.join(iter_dir, "models_saved")
        ensemble_dir = os.path.join(iter_dir, "ensemble")

        ev = IterationEval(
            hash_id=h,
            iter_dir=iter_dir,
            preds_dir=preds_dir,
            models_dir=models_dir,
            ensemble_dir=ensemble_dir
        )

        # Classifier metadata
        algo, model_bases = self._load_classifier_meta(ensemble_dir)
        ev.algo = algo
        ev.model_bases = model_bases

        # Classifier NPZ (never generated if missing)
        if algo:
            cls_npz = os.path.join(preds_dir, f"classifier_{algo}.npz")
            cls_json = os.path.join(preds_dir, f"classifier_{algo}_metrics.json")
            if os.path.exists(cls_npz):
                zc = _safe_np_load(cls_npz)
                if zc is not None:
                    ev.dates = pd.to_datetime(zc["dates"])
                    ev.y_true = zc["y_true"].astype(float)
                    ev.y_cls = zc["y_pred"].astype(float)
            if os.path.exists(cls_json):
                ev.metrics_classifier = _read_json(cls_json)

        # Per-model predictions for the band and the baseline
        # Pattern: pred_{lossname}_{pen}.npz (no _train/_test suffix)
        model_npz = sorted([p for p in glob.glob(os.path.join(preds_dir, "pred_*.npz"))
                            if not os.path.basename(p).startswith("classifier_")
                            and not os.path.basename(p).endswith("_train.npz")])
        if not model_npz:
            self.logger.warning("[EVAL] (%s) No pred_*.npz in %s", h, preds_dir)
            # Continue: there may be only a classifier
        band_preds = []
        band_dates = None
        baseline = None

        for p in model_npz:
            z = _safe_np_load(p)
            if z is None:
                continue
            base = _canonical_base(os.path.splitext(os.path.basename(p))[0].replace("pred_", ""))
            yp = z["y_pred"].astype(float)
            dts = pd.to_datetime(z["dates"])
            # align dates with the first file seen
            if band_dates is None:
                band_dates = dts
            else:
                if len(dts) != len(band_dates) or not np.all(dts.values == band_dates.values):
                    # date intersection, if any
                    common = np.intersect1d(band_dates.values, dts.values)
                    if common.size == 0:
                        self.logger.warning("[EVAL] (%s) No date intersection for the band. Skipping %s", h, p)
                        continue
                    # simple mask-based reindexing
                    mask0 = np.isin(band_dates.values, common)
                    mask1 = np.isin(dts.values, common)
                    band_dates = pd.to_datetime(common)
                    # crop everything accumulated so far
                    if band_preds:
                        band_preds = [bp[mask0] for bp in band_preds]
                    yp = yp[mask1]
            band_preds.append(yp)
            if base == "original_mae_0":
                baseline = yp

        if band_preds and band_dates is not None:
            stack = np.vstack(band_preds)  # (M, N)
            ev.band_min = np.min(stack, axis=0)
            ev.band_max = np.max(stack, axis=0)
            # without ev.dates (no classifier), use the band dates
            if ev.dates is None:
                ev.dates = band_dates
            # without the classifier y_true, take it from a model
            if ev.y_true is None and model_npz:
                z0 = _safe_np_load(model_npz[0])
                if z0 is not None:
                    ev.y_true = z0["y_true"].astype(float)
        else:
            self.logger.warning("[EVAL] (%s) Could not build the min–max band.", h)

        ev.y_mae = baseline  # may be None
        # Sanity: consistent shapes when dates and observations exist
        if ev.dates is not None and ev.y_true is not None:
            N = len(ev.dates)
            def _crop(arr):
                return arr[:N] if arr is not None and len(arr) != N else arr
            ev.y_cls = _crop(ev.y_cls)
            ev.y_mae = _crop(ev.y_mae)
            ev.band_min = _crop(ev.band_min)
            ev.band_max = _crop(ev.band_max)

        return ev

    def collect_all(self) -> List[IterationEval]:
        """Standardized data of every discovered iteration."""
        hashes = self.discover_iterations()
        out: List[IterationEval] = []
        for h in hashes:
            ev = self._collect_standard_data(h)
            if ev is None:
                continue
            out.append(ev)
        if not out:
            self.logger.warning("[EVAL] No iteration could be standardized.")
        return out

    # -------- Plots (read-only) --------

    def _compute_metrics_direct(self, y_true: np.ndarray, y_sel: np.ndarray, y_bas: Optional[np.ndarray],
                                band_min: Optional[np.ndarray], band_max: Optional[np.ndarray],
                                over: bool = True) -> Dict:
        """Simple metrics of a selected series; uses hydroeval when available."""
        out = {"overall": {}, "top": {}}
        m = ~(np.isnan(y_true) | np.isnan(y_sel))
        yt, ys = y_true[m], y_sel[m]
        if he is not None and yt.size and ys.size:
            try:
                out["overall"]["NSE"] = float(he.nse(ys, yt))
                out["overall"]["KGE"] = float(he.kge(ys, yt)[0][0])
            except Exception:
                pass

        # top/bottom 10%
        q = 0.90 if over else 0.10
        thr = float(np.quantile(yt, q))
        mask = yt >= thr if over else yt <= thr
        n = int(mask.sum())
        if n > 0:
            hits = int(np.sum(ys[mask] >= yt[mask])) if over else int(np.sum(ys[mask] <= yt[mask]))
            out["top"] = {
                "threshold": thr,
                "n": n,
                "hits": hits,
                "misses": int(n - hits),
                "hit_ratio": float(hits / n),
            }
        return out

    def plot_band_for_iteration(
        self,
        ev: IterationEval,
        save_subdir: str = "figs_band",
        filename_prefix: Optional[str] = None,
        title: Optional[str] = None,
        over: bool = True,
        # --- paper style ---
        font_base: int = 16,
        font_title: int = 20,
        dpi: int = 300,
        figsize: tuple = (14, 4.5),
    ) -> Optional[Dict]:
        """
        Paper band figure (concise labels, large fonts, no hash id in titles or filenames).
        """
        if ev.dates is None or ev.y_true is None or ev.band_min is None or ev.band_max is None or ev.y_cls is None:
            self.logger.warning("[EVAL] Missing data for the band plot (dates/y_true/band/cls).")
            return None

        # one folder per iteration, but NO hash in the file name
        save_dir = os.path.join(self.out_dir, save_subdir, ev.hash_id)
        _ensure_dir(save_dir)
        filename = (filename_prefix or ("band_over" if over else "band_low"))

        if not title:
            title = "Classifier vs penalized ensemble"

        # metrics
        metrics = self._compute_metrics_direct(ev.y_true, ev.y_cls, ev.y_mae, ev.band_min, ev.band_max, over=over)
        t   = pd.to_datetime(ev.dates)
        y_t = ev.y_true
        bmin, bmax = ev.band_min, ev.band_max
        y_cls = ev.y_cls
        y_base = ev.y_mae

        # top/bottom 10% threshold
        thr = float(np.percentile(y_t, 90.0 if over else 10.0))
        mask = (y_t >= thr) if over else (y_t <= thr)
        pct_label = "Top 10% observed" if over else "Bottom 10% observed"

        # large style
        plt.rcParams.update({
            "font.size": font_base,
            "axes.labelsize": font_base,
            "xtick.labelsize": font_base,
            "ytick.labelsize": font_base,
            "legend.fontsize": font_base,
            "axes.titlesize": font_title,
            "figure.dpi": 100,  # canvas dpi; exported at the higher 'dpi' below
        })

        # plot
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

        ax.fill_between(t, bmin, bmax, alpha=0.18, label="Ensemble range")
        ax.plot(t, y_t,   linewidth=1.2, alpha=0.7, linestyle="--", label="Observed")
        if y_base is not None:
            ax.plot(t, y_base, linewidth=1.4, alpha=0.9, linestyle="-.", label="Baseline")
        ax.plot(t, y_cls, linewidth=1.8, alpha=0.95, label=f"Classifier ({ev.algo or 'model'})")

        ax.axhline(thr, color="black", linewidth=1.0, alpha=0.25, linestyle=":", label=("P90 threshold" if over else "P10 threshold"))
        if np.any(mask):
            ax.scatter(t[mask], y_t[mask], s=18, marker="o", facecolor="none", edgecolor="red", linewidths=1.0, alpha=0.9, label=pct_label)

        ax.set_title(title)
        ax.set_ylabel("Inflow (m³/s)")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.legend(frameon=False, ncol=2)  # compact

        plt.tight_layout()

        png = os.path.join(save_dir, f"{filename}.png")  # NO hash in the file name
        pdf = os.path.join(save_dir, f"{filename}.pdf")
        plt.savefig(png, dpi=dpi)
        plt.savefig(pdf)
        plt.close()

        # CSV + metrics JSON
        csv = os.path.join(save_dir, f"{filename}.csv")
        df = pd.DataFrame({
            "Date": t, "Observed": y_t, "Band_min": bmin, "Band_max": bmax,
            "Classifier": y_cls, "TopMask": mask.astype(int),
            "Classifier_Hit": ((y_cls >= y_t) if over else (y_cls <= y_t)).astype(int),
        })
        if y_base is not None:
            df["Baseline"] = y_base
            df["Baseline_Hit"] = ((y_base >= y_t) if over else (y_base <= y_t)).astype(int)
        df.to_csv(csv, index=False)

        jsonp = os.path.join(save_dir, f"{filename}_metrics.json")
        with open(jsonp, "w") as f:
            json.dump(metrics, f, indent=2)

        self.logger.info("[EVAL] Saved band plot: %s / %s", png, pdf)
        return {"png": png, "pdf": pdf, "csv": csv, "metrics": jsonp, "metrics_vals": metrics}

    
    def plot_models_for_iteration(
        self,
        ev: IterationEval,
        *,
        save_subdir: str = "figs_models",
        title_prefix: str = "Observed, baseline, and penalized model",
        over: bool = True,
        include_only: Optional[List[str]] = None,
        # paper style
        font_base: int = 16,
        font_title: int = 20,
        dpi: int = 300,
        figsize: tuple = (14, 4.5),
    ) -> List[Dict]:
        """
        One figure per penalized model (large fonts, concise labels, shows Observed, Baseline, and Penalized).
        """
        if ev.dates is None or ev.y_true is None:
            self.logger.warning("[EVAL] No dates/y_true for per-model plots.")
            return []

        preds_dir = ev.preds_dir
        if not os.path.isdir(preds_dir):
            self.logger.warning("[EVAL] predictions_cache not found.")
            return []

        model_paths = sorted([
            p for p in glob.glob(os.path.join(preds_dir, "pred_*.npz"))
            if not os.path.basename(p).startswith("classifier_")
            and not os.path.basename(p).endswith("_train.npz")
        ])
        if not model_paths:
            self.logger.warning("[EVAL] No pred_*.npz found for per-model plots.")
            return []

        target_dates = pd.to_datetime(ev.dates)
        y_true = ev.y_true.astype(float)
        baseline = ev.y_mae

        per_model = []
        for p in model_paths:
            name = os.path.splitext(os.path.basename(p))[0].replace("pred_", "")
            base = _canonical_base(name)

            if base == "original_mae_0":
                continue
            if include_only is not None and base not in include_only:
                continue

            z = _safe_np_load(p)
            if z is None:
                continue

            dts = pd.to_datetime(z["dates"])
            yp = z["y_pred"].astype(float)

            # Align dates
            if len(dts) != len(target_dates) or not np.all(dts.values == target_dates.values):
                common = np.intersect1d(target_dates.values, dts.values)
                if common.size == 0:
                    self.logger.warning("[EVAL] %s has no overlapping dates. Skipped.", base)
                    continue
                mask_t = np.isin(target_dates.values, common)
                mask_m = np.isin(dts.values, common)
                dts_use = pd.to_datetime(common)
                y_true_use = y_true[mask_t]
                yp_use = yp[mask_m]
                baseline_use = baseline[mask_t] if baseline is not None else None
            else:
                dts_use = target_dates
                y_true_use = y_true
                yp_use = yp
                baseline_use = baseline

            # Top/Bottom 10 %
            thr = float(np.percentile(y_true_use, 90.0 if over else 10.0))
            mask_pct = (y_true_use >= thr) if over else (y_true_use <= thr)

            # Extract only penalty index (last numeric part)
            import re
            match = re.search(r"(\d+(?:\.\d+)?)$", base)
            penalty_label = match.group(1) if match else base

            # Quick metrics
            metrics = self._compute_metrics_direct(
                y_true=y_true_use,
                y_sel=yp_use,
                y_bas=baseline_use,
                band_min=None, band_max=None,
                over=over
            )

            # Save paths
            save_dir = os.path.join(self.out_dir, save_subdir, ev.hash_id)
            _ensure_dir(save_dir)
            fname = f"model_penalty_{penalty_label}" + ("_over" if over else "_low")
            png = os.path.join(save_dir, f"{fname}.png")
            pdf = os.path.join(save_dir, f"{fname}.pdf")
            csv = os.path.join(save_dir, f"{fname}.csv")
            jsonp = os.path.join(save_dir, f"{fname}_metrics.json")

            # Style
            plt.rcParams.update({
                "font.size": font_base,
                "axes.labelsize": font_base,
                "xtick.labelsize": font_base,
                "ytick.labelsize": font_base,
                "legend.fontsize": font_base,
                "axes.titlesize": font_title,
                "figure.dpi": 100,
            })

            # Plot
            plt.figure(figsize=figsize)
            plt.plot(dts_use, y_true_use, linewidth=1.4, alpha=0.75, linestyle="--", label="Observed")
            if baseline_use is not None:
                plt.plot(dts_use, baseline_use, linewidth=1.6, alpha=0.9, linestyle="-.", label="Baseline")
            plt.plot(dts_use, yp_use, linewidth=1.8, alpha=0.95, label="Penalized model")

            plt.axhline(thr, color="black", linewidth=1.0, alpha=0.25, linestyle=":",
                        label=("P90 threshold" if over else "P10 threshold"))
            if np.any(mask_pct):
                plt.scatter(dts_use[mask_pct], y_true_use[mask_pct], s=18, marker="o",
                            facecolor="none", edgecolor="red", linewidths=1.0, alpha=0.9,
                            label=("Top 10% observed" if over else "Bottom 10% observed"))

            # ---- Short, informative title ----
            ttl = f"{title_prefix} (penalization coef = {penalty_label})"
            plt.title(ttl)
            plt.ylabel("Inflow (m³/s)")
            plt.xlabel("Date")
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.grid(alpha=0.25, linewidth=0.8)
            plt.legend(frameon=False, ncol=2)
            plt.tight_layout()
            plt.savefig(png, dpi=dpi)
            plt.savefig(pdf)
            plt.close()

            # CSV
            df_out = pd.DataFrame({
                "Date": dts_use,
                "Observed": y_true_use,
                "Penalized": yp_use,
                "TopMask": mask_pct.astype(int),
                "Hit": ((yp_use >= y_true_use) if over else (yp_use <= y_true_use)).astype(int),
            })
            if baseline_use is not None:
                df_out["Baseline"] = baseline_use
                df_out["Baseline_Hit"] = ((baseline_use >= y_true_use) if over else (baseline_use <= y_true_use)).astype(int)
            df_out.to_csv(csv, index=False)

            with open(jsonp, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            self.logger.info("[EVAL] Saved penalized plot %s (coef=%s)", base, penalty_label)
            per_model.append({
                "model": base,
                "penalty_coef": penalty_label,
                "png": png, "pdf": pdf, "csv": csv,
                "metrics": jsonp, "metrics_vals": metrics
            })

        if not per_model:
            self.logger.warning("[EVAL] No per-model plots generated.")
        return per_model




    # -------- Metric tables --------

    def classifier_metrics_table(self, iterations: List[IterationEval], save: bool = True,
                                 filename: str = "classifier_metrics_summary.csv") -> pd.DataFrame:
        """Table with one row per iteration (classifier) and one column per metric.

        Appends ``mean``, ``std`` and ``ci_95_margin`` (95% confidence interval
        half-width) rows.

        Args:
            iterations (List[IterationEval]): Iterations.
            save (bool): Write the CSV (and the per-algorithm summary).
            filename (str): Output file name.

        Returns:
            pd.DataFrame: The table (empty if no classifier metrics exist).
        """
        rows = []
        for ev in iterations:
            if not ev.metrics_classifier:
                self.logger.warning("[EVAL] (%s) No classifier metrics. Omitted from the table.", ev.hash_id)
                continue
            
            met = ev.metrics_classifier
            
            # Extended metric collection
            row = {
                "hash": ev.hash_id,
                "algo": ev.algo,
                # Classic metrics (old and new formats supported)
                "overall_NSE": met.get("global_nse", met.get("overall", {}).get("NSE")),
                "overall_KGE": met.get("global_kge", met.get("overall", {}).get("KGE")),
                
                # Operational and false-alarm metrics
                "Precision": met.get("Precision"),
                "Recall_HitRatio": met.get("Recall_HitRatio", met.get("top10", {}).get("hit_ratio")),
                "FAR": met.get("FAR"),
                "F1_Score": met.get("F1_Score"),
                
                # Timing error in days
                "Timing_Error_Days": met.get("mean_absolute_timing_error"),
                
                # Error restricted to extremes (training threshold)
                "Extremes_MAE": met.get("mae_extremes"),
                "Extremes_MSE": met.get("mse_extremes"),
            }
            
            # Drop None values (a metric may have failed for a given year)
            row = {k: v for k, v in row.items() if v is not None}
            rows.append(row)

        if not rows:
            self.logger.warning("[EVAL] No classifier metric rows to tabulate.")
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("hash")

        # Summary statistics across iterations
        numeric = df.select_dtypes(include=[float, int]).columns
        n_years = len(df) # N (number of iterations/years)
        
        # 1. Mean
        mean_row = df[numeric].mean().rename("mean")
        # 2. Standard deviation
        std_row  = df[numeric].std(ddof=1 if n_years > 1 else 0).rename("std")
        # 3. 95% confidence interval (Z ≈ 1.96) -> ± margin
        ci_95_margin = (1.96 * (std_row / np.sqrt(n_years))).rename("ci_95_margin")
        
        df_out = pd.concat([df, pd.DataFrame([mean_row, std_row, ci_95_margin])])

        # Optional grouping by algorithm
        if "algo" in df.columns:
            by_algo = df.groupby("algo")[numeric].agg(["mean", "std"])
            by_algo.columns = [f"{c[0]}_{c[1]}" for c in by_algo.columns]
            algo_csv = os.path.join(self.out_dir, filename.replace(".csv", "_by_algo.csv"))
            by_algo.to_csv(algo_csv, index=True)
            self.logger.info("[EVAL] Saved per-algorithm summary: %s", algo_csv)

        if save:
            out_csv = os.path.join(self.out_dir, filename)
            df_out.to_csv(out_csv, index=True)
            self.logger.info("[EVAL] Saved metrics summary with confidence intervals to: %s", out_csv)
            
        return df_out

    # -------- All-in-one (read-only) evaluation --------

    def run(self, make_plots: bool = True, over: bool = True) -> Dict[str, str]:
        """Load every iteration, create the band plots and the classifier metrics table.

        Missing artifacts are reported, never generated.

        Args:
            make_plots (bool): Create the plots.
            over (bool): Evaluate the upper (top 10%) rather than the lower tail.

        Returns:
            Dict[str, str]: Output paths.
        """
        iters = self.collect_all()
        outputs = {}

        # Plots
        if make_plots:
            for ev in iters:
                self.plot_band_for_iteration(ev, save_subdir="figs_band", filename_prefix="band_over_paper",
                                             title=f"Observed, baseline, classifier, and penalized ensemble range", over=over)
                self.plot_models_for_iteration(
                    ev,
                    save_subdir="figs_models_over" if over else "figs_models_low",
                    title_prefix="Penalized model vs baseline",
                    over=over
                )


        # Classifier metrics table
        df = self.classifier_metrics_table(iters, save=True, filename="classifier_metrics_summary.csv")
        if not df.empty:
            outputs["metrics_csv"] = os.path.join(self.out_dir, "classifier_metrics_summary.csv")

        return outputs


# ----------------------------- Public function -------------------------------

def generate_global_report(numero_prueba: str,
                           result_root: Optional[str] = None,
                           make_plots: bool = True,
                           over: bool = True) -> Dict[str, str]:
    """Top-level entry point: evaluation report over every iteration of an experiment.

    Collects the iterations under ``tests{numero_prueba}``, creates the band
    plots (when data exists) and the classifier metrics table with means and
    standard deviations. Missing artifacts are never generated.

    Args:
        numero_prueba (str): Experiment number.
        result_root (Optional[str]): Results root.
        make_plots (bool): Create the plots.
        over (bool): Evaluate the upper tail.

    Returns:
        Dict[str, str]: Output paths.
    """
    ev = ExperimentEvaluator(numero_prueba=numero_prueba, result_root=result_root)
    return ev.run(make_plots=make_plots, over=over)
