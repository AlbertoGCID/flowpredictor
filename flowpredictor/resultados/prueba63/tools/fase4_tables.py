"""Sensitivity tables (Markdown) built from results_consolidated.csv.

Five tables: the audited CV table with paired Wilcoxon tests, baselines at
tau=0.50 vs tau=0.90, p90 vs p95 threshold, stability across seeds, and the
selection of M* on pure cross-validation.

Usage:  python3 resultados/prueba63/tools/fase4_tables.py
Output: stdout and resultados/prueba63/doc/fase4_tables.md
"""
from typing import Any, Dict, Optional

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # .../flowpredictor
sys.path.insert(0, str(ROOT))
from resultados.prueba63.src.evaluation.model_selection import (  # noqa: E402
    aggregate_config_metrics, lexicographic_constrained_score, paired_fold_tests,
)

df = pd.read_csv(ROOT / "resultados/prueba63/results_consolidated.csv")
df = df[df["horizon"] == "48h"]
df["seed"] = df["seed"].fillna(92).astype(int)
df["threshold_percentile"] = df["threshold_percentile"].fillna(90).astype(int)
out = []


def emit(s: str = "") -> None:
    """Print a line and append it to the Markdown output."""
    print(s)
    out.append(s)


def ms(x: pd.Series, digits: int = 3) -> str:
    """Format ``mean ± std`` (or the mean alone for a single value)."""
    x = x.dropna()
    if x.empty:
        return "—"
    return f"{x.mean():.{digits}f} ± {x.std(ddof=1):.{digits}f}" if len(x) > 1 else f"{x.mean():.{digits}f}"


def mean(x: pd.Series, digits: int = 3) -> str:
    """Format the mean of a series."""
    x = x.dropna()
    return f"{x.mean():.{digits}f}" if not x.empty else "—"


def pval(p: Optional[float]) -> str:
    """Format a p-value."""
    return "—" if p is None else (f"{p:.4f}" if p >= 1e-4 else "<0.0001")


def pcell(res: Optional[Dict[str, Any]]) -> str:
    """Wilcoxon p-value with the sign and size of the mean difference vs. M1.

    A small p alone does not say whether the configuration improves or worsens.
    """
    if not res or res.get("mean_diff") is None:
        return "—"
    mark = "" if res.get("p_exact") else "≈"
    return f"{mark}{pval(res['wilcoxon_p'])} (Δ {res['mean_diff']:+.3f})"


def variant(seed: int = 92, pct: int = 90, strategy: str = "expanding") -> pd.DataFrame:
    """Rows of one seed / threshold percentile / strategy."""
    return df[(df["seed"] == seed) & (df["threshold_percentile"] == pct) & (df["strategy"] == strategy)]


# ---------------------------------------------------------------- Table 1
# Canonical tau=0.50 baselines only: the tau=0.90 variants are reserved for the
# comparative Table 2.
T1 = ["M1", "M2", "M5", "heuristic_ensemble", "quantile_regression", "gradient_boosting_quantile"]
base = variant()
tests = paired_fold_tests(df, reference="M1", configs=[c for c in T1 if c != "M1"],
                          metrics=("NSE", "HitRatio"), horizon="48h", strategy="expanding")
emit("## Tabla 1 auditada — CV expansiva 2017–2023 · 48 h · seed 92 · p90")
emit("")
emit("| Config | n | NSE (media±std) | KGE (media±std) | HitRatio (media±std) | FARate | FARatio | F1 | Signed Lag (d) | Absolute Lag (d) | Wilcoxon p NSE vs M1 | Wilcoxon p Hit vs M1 |")
emit("|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in T1:
    s = base[base["config"] == c]
    if s.empty:
        emit(f"| {c} | 0 | — | — | — | — | — | — | — | — | — | — |")
        continue
    t = tests.get(c, {})
    p_nse = "ref." if c == "M1" else pcell(t.get("NSE"))
    p_hit = "ref." if c == "M1" else pcell(t.get("HitRatio"))
    emit(f"| {c} | {len(s)} | {ms(s['NSE'])} | {ms(s['KGE'])} | {ms(s['HitRatio'])} | {mean(s['FARate'])} | "
         f"{mean(s['FARatio'])} | {mean(s['F1'])} | {mean(s['PeakTiming_mean_lag'])} | "
         f"{mean(s['PeakTiming_mean_absolute_lag'])} | {p_nse} | {p_hit} |")
emit("")
emit("Wilcoxon de rangos con signo pareado por año de test (7 folds); Δ = media entre folds de (config − M1). "
     "Sin diferencias nulas, con n=7 el p exacto mínimo es 0.0156. «≈»: algún fold con diferencia exactamente "
     "cero; scipy recurre a la aproximación normal, poco fiable con n=7.")

# ---------------------------------------------------------------- Table 2
emit("")
emit("## Tabla 2 — Baselines τ=0.50 vs τ=0.90 · 48 h · seed 92 · p90")
emit("")
emit("| Baseline | τ | Estrategia | n | NSE | HitRatio | Recall | FARate |")
emit("|---|---|---|---|---|---|---|---|")
for name, t50, t90 in (("Quantile Regression", "quantile_regression", "quantile_regression_tau90"),
                       ("Gradient Boosting", "gradient_boosting_quantile", "gradient_boosting_tau90")):
    for strategy in ("expanding", "holdout_final"):
        for tau, cfg in (("0.50", t50), ("0.90", t90)):
            s = variant(strategy=strategy)
            s = s[s["config"] == cfg]
            emit(f"| {name} | {tau} | {strategy} | {len(s)} | {ms(s['NSE'])} | {ms(s['HitRatio'])} | "
                 f"{ms(s['Recall'])} | {ms(s['FARate'])} |")

# ---------------------------------------------------------------- Table 3
n_col = next((c for c in ("n_extreme", "Extreme_n_extreme", "n_events") if c in df.columns), None)
emit("")
emit("## Tabla 3 — Sensibilidad de umbral p90 vs p95 · 48 h · seed 92")
emit("")
emit("| Config | Estrategia | Umbral | n | HitRatio | FARate | Recall | N eventos (total) |")
emit("|---|---|---|---|---|---|---|---|")
for c in ("M1", "M2", "M5"):
    for strategy in ("expanding", "holdout_final"):
        for pct in (90, 95):
            s = variant(pct=pct, strategy=strategy)
            s = s[s["config"] == c]
            n_ev = int(s[n_col].sum()) if n_col and not s.empty else "—"
            emit(f"| {c} | {strategy} | p{pct} | {len(s)} | {ms(s['HitRatio'])} | {ms(s['FARate'])} | "
                 f"{ms(s['Recall'])} | {n_ev} |")
if n_col is None:
    emit("")
    emit("Aviso: el CSV no contiene columna de número de eventos extremos; columna N eventos vacía.")

# ---------------------------------------------------------------- Table 4
emit("")
emit("## Tabla 4 — Estabilidad por semillas (92, 42, 123) · p90 · 48 h")
emit("")
emit("Por semilla se toma la media sobre los 7 folds de CV expansiva; la tabla resume esas medias entre semillas. El holdout (1 valor por semilla) va aparte.")
emit("")
emit("Una semilla solo cuenta si tiene el conjunto COMPLETO de folds (7 expanding / 1 holdout); las parciales se listan aparte y no entran en los estadísticos.")
emit("")
emit("| Config | Métrica | Estrategia | Semillas completas | Media | Min | Max | Std | Incompletas (folds) |")
emit("|---|---|---|---|---|---|---|---|---|")
EXPECTED_FOLDS = {"expanding": 7, "holdout_final": 1}
for c in ("M2", "M5"):
    for metric in ("NSE", "HitRatio"):
        for strategy in ("expanding", "holdout_final"):
            per_seed, incomplete = {}, []
            for seed in (92, 42, 123):
                s = variant(seed=seed, strategy=strategy)
                s = s[s["config"] == c][metric].dropna()
                if s.empty:
                    continue
                if len(s) < EXPECTED_FOLDS[strategy]:
                    incomplete.append(f"{seed} ({len(s)}/{EXPECTED_FOLDS[strategy]})")
                    continue
                per_seed[seed] = s.mean()
            inc = ", ".join(incomplete) or "—"
            v = np.array(list(per_seed.values()))
            if v.size == 0:
                emit(f"| {c} | {metric} | {strategy} | 0 | — | — | — | — | {inc} |")
                continue
            std = f"{v.std(ddof=1):.3f}" if v.size > 1 else "—"
            emit(f"| {c} | {metric} | {strategy} | {sorted(per_seed)} | {v.mean():.3f} | {v.min():.3f} | "
                 f"{v.max():.3f} | {std} | {inc} |")

# ---------------------------------------------------------------- Table 5
full = pd.read_csv(ROOT / "resultados/prueba63/results_consolidated.csv")
agg = aggregate_config_metrics(full, horizon="48h", exclude_holdout=True)
emit("")
emit("## Tabla 5 — Selección de M* recalculada · CV pura (expanding + LOYO, sin holdout) · seed 92 · p90")
emit("")
emit("| Config | n folds | NSE | FARate | ΔHitRatio | α ≤ 0.10 | α ≤ 0.12 |")
emit("|---|---|---|---|---|---|---|")
scores = {a: lexicographic_constrained_score(agg, farate_max=a) for a in (0.10, 0.12)}
for c in ("M1", "M2", "M3", "M4", "M5"):
    m = agg[c]["combined"]
    cell = {a: ("excluida" if scores[a][c] == float("-inf") else f"{scores[a][c]:.4f}") for a in scores}
    emit(f"| {c} | {m['n_folds']} | {m['NSE_mean']:.4f} | {m['FARate_mean']:.4f} | "
         f"{m['DeltaHitRatio_mean']:.4f} | {cell[0.10]} | {cell[0.12]} |")
for a in (0.10, 0.12):
    emit(f"\n**M\\* con α ≤ {a:.2f}: {max(scores[a], key=scores[a].get)}**")

out_path = ROOT / "resultados/prueba63/doc/fase4_tables.md"
out_path.write_text("\n".join(out), encoding="utf-8")
print(f"\n[fase4_tables] saved to {out_path}")
