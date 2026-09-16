"""Reproducibility and non-regression gate for the published results.

  [3] Always: the LaTeX tables of the manuscript are regenerated from
      results_consolidated.csv (in a temporary directory; doc/tables_latex.tex
      is never overwritten) and must be identical to doc/tables_latex.tex.
  [1] If a backup CSV is available: every backup row is still present with
      identical metrics.
  [2] If a backup CSV is available: new rows are only sensitivity variants
      (_p<pct>/_s<seed>) or tau90 baselines.

Exits with status 1 if any executed check fails. No model is retrained.

Usage:
  python3 resultados/prueba63/tools/verify_published.py [--backup CSV_PATH]
"""
import argparse
import difflib
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # resultados/prueba63
sys.path.insert(0, str(ROOT.parents[1]))  # project root

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--backup", type=Path, default=ROOT / "results_consolidated_pre_sensitivity_backup.csv")
args = p.parse_args()

current = pd.read_csv(ROOT / "results_consolidated.csv")
failed = False

if args.backup.exists():
    backup = pd.read_csv(args.backup)
    print(f"backup rows={len(backup)}  current rows={len(current)}")
    for col in ("seed", "threshold_percentile"):
        print(f"column {col}: present={col in current.columns}  NaN={current[col].isna().sum() if col in current else 'n/a'}")

    cur = current.set_index("fold")
    metric_cols = [c for c in backup.columns if c != "fold" and pd.api.types.is_numeric_dtype(backup[c])]
    missing, changed = [], []
    for _, row in backup.iterrows():
        f = row["fold"]
        if f not in cur.index:
            missing.append(f)
            continue
        new = cur.loc[f]
        for c in metric_cols:
            a, b = row[c], new[c]
            if (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and np.isclose(a, b, rtol=0, atol=1e-12)):
                continue
            changed.append((f, c, a, b))

    print(f"\n[1] backup rows missing: {len(missing)}  {missing[:5]}")
    print(f"[1] backup values changed: {len(changed)}")
    for f, c, a, b in changed[:15]:
        print(f"      {f} · {c}: {a} -> {b}")

    variant_mark = re.compile(r"_p\d+|_s\d+|tau90")
    new_folds = sorted(set(current["fold"]) - set(backup["fold"]))
    unexpected = [f for f in new_folds if not variant_mark.search(f)]
    print(f"\n[2] new rows: {len(new_folds)}  unexpected (no variant mark): {len(unexpected)} {unexpected[:5]}")
    if new_folds:
        print(current[current["fold"].isin(new_folds)].groupby(["config", "strategy"], dropna=False).size().to_string())
    failed = failed or bool(missing or changed or unexpected)
else:
    print(f"[1-2] skipped: backup CSV not found ({args.backup.name}); rows in results_consolidated.csv: {len(current)}")

import resultados.prueba63.src.visualization.generate_paper_figures as G  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="tables_check_"))
G.export_latex_tables(G.load_results(), doc_dir=tmp)
reference = ROOT / "doc" / "tables_latex.tex"
if not reference.exists():
    print(f"\n[3] FAILED: reference table file not found: {reference}")
    sys.exit(1)
diff = list(difflib.unified_diff(reference.read_text().splitlines(),
                                 (tmp / "tables_latex.tex").read_text().splitlines(), lineterm=""))
print(f"\n[3] regenerated tables vs doc/tables_latex.tex: {'IDENTICAL' if not diff else f'{len(diff)} diff lines'}")
print("\n".join(diff[:40]))
failed = failed or bool(diff)

print(f"\n[verify_published] {'FAILED' if failed else 'PASSED'}")
sys.exit(1 if failed else 0)
