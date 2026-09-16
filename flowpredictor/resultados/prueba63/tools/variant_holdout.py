"""2025 holdout (48h) of a seed variant, reusing the pipeline functions unchanged.

``--run_holdout_final`` only evaluates M*, M1 and the baselines; the seed
sensitivity analysis needs M1, M2 and M5 explicitly.

Usage:
  python3 resultados/prueba63/tools/variant_holdout.py --seed 42 --configs M1 M2 M5
  python3 resultados/prueba63/tools/variant_holdout.py --seed 92 --baselines
"""
import argparse
import logging
import os
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--configs", nargs="*", default=[])
p.add_argument("--baselines", action="store_true")
a = p.parse_args()

_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../flowpredictor
sys.path.insert(0, str(_REPO_ROOT))
# dataset.root_raw is relative to the parent directory of flowpredictor: if a bundle
# has to be regenerated (new hash), the cwd must be that directory.
os.chdir(_REPO_ROOT.parent)

from resultados.prueba63.log.log_config import init_logger, get_logger  # noqa: E402
import resultados.prueba63.src.run_experiments_phase3 as R  # noqa: E402

R.BASE_ITERATION_DEFAULTS["seed"] = a.seed
init_logger(logging.INFO, numero_prueba=R.NUMERO_PRUEBA)
logger = get_logger()
assert "_smoke" not in str(R.RESULTS_CSV) and "_e2e_verify" not in str(R.RESULTS_CSV), R.RESULTS_CSV

logger.info("[variant_holdout] seed=%s suffix=%r configs=%s baselines=%s",
            a.seed, R._variant_suffix(), a.configs, a.baselines)
for cfg in a.configs:
    row = R.run_holdout_evaluation(cfg, logger, resume=True)
    logger.info("[variant_holdout] %s -> %s", cfg, "new row" if row else "no new row (already complete or failed)")
if a.baselines:
    rows = R.run_holdout_baselines(logger, resume=True)
    logger.info("[variant_holdout] holdout baselines -> %d rows", len(rows))
