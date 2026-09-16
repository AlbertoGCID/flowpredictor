# Flowpredictor

Safety-biased multi-step reservoir inflow forecasting with asymmetric Seq2Seq
LSTMs and dynamic model routing.

This repository contains the code, configuration and test suite behind the
experiments reported in the accompanying manuscript. It forecasts daily
reservoir inflow (`Qe`, m³/s) at 24 h, 48 h and 72 h lead times from upstream
rain gauges and numerical precipitation forecasts. It is built for
flood-safety operation, where under-predicting a flood peak costs much more
than over-predicting it.

---

## 1. Scientific overview

### 1.1 Seq2Seq encoder–decoder
- **Encoder:** an LSTM over a 30-day context window of historical inflow and
  rain-gauge records.
- **Decoder:** an autoregressive LSTM that unrolls `offset + 1` daily steps.
  At each step it receives the previous inflow prediction and the matching
  daily precipitation forecast (24 h, 48 h, 72 h). The evaluated horizon is
  the **last** decoder step.
- **State passing:** the encoder's hidden and cell states initialize the
  decoder directly. No Dense layer sits in the recurrent path, so the
  LSTM's constant error carousel is preserved. A linear state projector is
  used only if the encoder and decoder widths differ; in the published
  configurations both are 32 units, so it is never built.
- **Rainfall-only pretraining (optional):** models can be pretrained with the
  inflow channel masked to zero, then fine-tuned on the full inputs.

### 1.2 Asymmetric loss
Training uses a pinball (quantile) loss whose quantile is derived from an
integer penalty `p`:

```
tau = clip((p + 1) / (p + 2), 0.5, 0.99)        e.g. p=2 -> tau=0.75
L   = mean(max(tau * e, (tau - 1) * e)),  e = y_true - y_pred
```

`p = 0` gives the symmetric MAE baseline. Larger penalties make
under-prediction more expensive and shift forecasts towards the upper
quantiles of the inflow distribution.

### 1.3 Dual operating regime
The ablation study defines five configurations (`experiments_config.toml`,
`[ablation_variants.*]`):

| Config | Pretraining | Loss | Penalty grid | RF routing |
|---|---|---|---|---|
| M1 | no  | symmetric MAE | {0} | no |
| M2 | no  | pinball | {2} | no |
| M3 | yes | symmetric MAE | {0} | no |
| M4 | yes | pinball | {2} | no |
| M5 | yes | pinball | {2, 4, 6, 8, 10} | yes |

The framework runs in two regimes:
- **Standard regime:** one Seq2Seq model with a fixed moderate asymmetric
  penalty (M2/M4). It keeps a good global hydrograph fit (NSE/KGE) and a low
  false alarm rate.
- **Emergency regime (M5):** one Seq2Seq branch is trained per penalty. A
  Random Forest meta-learner then chooses, for each time step, which branch
  to serve. The meta-learner is trained on **nested, blocked out-of-fold**
  predictions, and its target is the branch that minimizes the asymmetric
  loss at the grid's maximum penalty. This regime favors early peak
  interception and trades away some global fit.

### 1.4 Evaluation protocol
- **Expanding-window cross-validation:** seven hydrological-year folds
  (July–June), test years 2017–2023. Each fold trains only on data before its
  test year.
- **Leave-one-year-out CV** (optional, `--run_cv_loyo`/`--run_ablation_loyo`).
- **Blind holdout:** 1 July 2024 onwards (label `2025`). It is evaluated once,
  on the configuration chosen by a constrained lexicographic rule: maximize
  ΔHit Ratio subject to NSE > 0 and false alarm rate ≤ 0.10.
- **Anti-leakage:** normalization parameters and the extreme-event threshold
  (p90 by default) come from the training partition only, excluding imputed
  rows. Computing either on test data raises an error.
- **Metrics:** NSE, KGE, Hit Ratio, false alarm rate/ratio, F1, precision,
  recall, signed and absolute peak-timing error, and bootstrap 95% confidence
  intervals.
- **Robustness:** Monte Carlo rainfall perturbation (uniform ±10%, Gaussian
  ±5%, 50 replicas) applied to the selected model.
- **Baselines:** linear quantile regression and quantile gradient boosting
  (τ = 0.50 and τ = 0.90), plus an inverse-MAE weighted ensemble fitted on
  training predictions only.

---

## 2. Repository layout

```
flowpredictor/
├── requirements.txt
├── tests/                              # pytest suite (leakage, CV splits, metrics, ...)
└── resultados/prueba63/
    ├── experiments_config.toml         # single source of paths and hyperparameters
    ├── log/                            # logger configuration
    ├── src/
    │   ├── run_experiments_phase3.py   # experiment orchestrator (CLI)
    │   ├── main_pipeline.py            # ablation configs, CV plan, baseline runner
    │   ├── config.py, config_schema.py # TOML loading and validation
    │   ├── data/dataset_generator.py   # gap filling, splits, normalized bundles
    │   ├── pipeline/                   # Iteration (training/inference/caching), windows, I/O
    │   ├── models/                     # Seq2Seq, losses, training loop, RF/XGBoost, baselines
    │   ├── evaluation/                 # metrics, model selection, perturbation test
    │   └── visualization/              # paper figures, LaTeX tables, yearly hydrographs
    └── tools/                          # sensitivity experiments (seeds, p95, tau=0.90)
```

---

## 3. Environment

The published results were produced with **Python 3.10.12** on Linux.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r flowpredictor/requirements.txt
```

TensorFlow 2.15.1 runs on CPU or GPU. On GPU, training can differ at the
floating-point level from run to run even with fixed seeds, because some cuDNN
kernels are non-deterministic.

Seeds: the published run uses seed `92` for training, data imputation and the
robustness replicas (`experiments_config.toml`).

---

## 4. Data

The raw hydrometeorological dataset is **not redistributed** in this
repository. The pipeline expects a single daily CSV at the path set in
`experiments_config.toml` → `[dataset].root_raw`
(default `flowpredictor/datasets/dataset_completo_2009_2025.csv`, relative to the
directory that contains `flowpredictor/`), with these columns:

| Column | Description |
|---|---|
| `Fecha` | date (daily) |
| `Qe` | reservoir inflow (m³/s), prediction target |
| `l/m2_arzua`, `l/m2_serradofaro`, `l/m2_melide`, `l/m2_olveda` | observed daily rainfall at the upstream gauges (l/m²) |
| `pred_l/m2`, `pred_l/m2_2d`, `pred_l/m2_3d` | numerical precipitation forecasts for the next 24 h, 48 h and 72 h |

`src/data/dataset_generator.py` fills date gaps deterministically, flags the
filled rows with `is_imputed`, and writes one normalized bundle per split and
test year:

```
datasets/normalizados/<split>/test<YEAR>/
    normalized/train.csv, normalized/test.csv
    normalization_params.json          # min/range and p90/p95 of Qe, computed on train only
```

Split labels: `JunioExpanding` (expanding window and holdout) and `Junio`
(LOYO). The bundle builders are `run_expanding_window`, `run_holdout_bundle`
and `run`.

---

## 5. Reproducing the experiments

Run every command from the **directory that contains `flowpredictor/`**, with
`flowpredictor/` on `PYTHONPATH`. Dataset paths in the configuration are relative
to that directory.

```bash
export PYTHONPATH="$PWD/flowpredictor"
MOD=resultados.prueba63.src.run_experiments_phase3
```

Long runs should be started inside `tmux` with the output sent to a log file.
Every fold is checkpointed atomically, and a re-run resumes where it stopped.
Pass `--no_resume` to recompute.

### 5.1 Cross-validation (M5 + baselines)
```bash
python3 -m $MOD --run_cv
```

### 5.2 Ablation study (M1–M4)
```bash
python3 -m $MOD --run_ablation
python3 -m $MOD --run_ablation --configs M1 M2        # subset of configurations
```

### 5.3 Blind holdout (selected model, M1 and baselines)
```bash
python3 -m $MOD --run_holdout_final
```

### 5.4 Robustness to rainfall-forecast perturbations
```bash
python3 -m $MOD --run_robustness                      # default: holdout year 2025, 50 replicas
python3 -m $MOD --run_robustness --n_replicas 50 --robustness_years 2025
```

### 5.5 Everything (CV + ablation + robustness)
```bash
python3 -m $MOD --all
```

### 5.6 Other options
| Flag | Effect |
|---|---|
| `--offset {0,1,2}` | evaluated horizon: 24 h, 48 h (default), 72 h |
| `--seed N` | training seed (default 92); other seeds use a separate cache and `_s<N>` fold labels |
| `--threshold_percentile {90,95}` | extreme-event threshold percentile (default 90) |
| `--recalculate_metrics` | recompute metrics from saved predictions with the training threshold, no retraining |
| `--run_cv_loyo`, `--run_ablation_loyo`, `--loyo_years` | leave-one-year-out variants |
| `--years`, `--penalty`, `--max_epochs`, `--max_epochs_clasificador`, `--early_stopping_patience` | restrict years or override hyperparameters |

### 5.7 Sensitivity experiments
```bash
bash flowpredictor/resultados/prueba63/tools/run_sensitivity.sh                  # seeds 42/123, tau=0.90 baselines
python3 -m $MOD --recalculate_metrics --threshold_percentile 95               # p95 threshold, no retraining
python3 flowpredictor/resultados/prueba63/tools/variant_holdout.py --seed 42 --configs M1 M2 M5
python3 flowpredictor/resultados/prueba63/tools/fase4_tables.py                  # sensitivity tables + Wilcoxon tests
python3 flowpredictor/resultados/prueba63/tools/verify_published.py              # non-regression check of published rows
```

### 5.8 Figures and tables
These read the saved results and predictions only. They never retrain.
```bash
python3 -m resultados.prueba63.src.visualization.generate_paper_figures --metric NSE
python3 -m resultados.prueba63.src.visualization.generate_yearly_plots
```

### 5.9 Tests
```bash
cd flowpredictor && python3 -m pytest
```
The integrity tests in `tests/test_results_integrity.py` are skipped until
`results_consolidated.csv` exists.

---

## 6. Outputs

All paths are relative to `flowpredictor/resultados/prueba63/`.

| Path | Content |
|---|---|
| `results_consolidated.csv` | one row per fold, keyed by `fold` (e.g. `M5_2019`, `M2_holdout_final`, `robustness_M5_uniform_2025`): `config`, `strategy`, `horizon`, `test_year`, `hash`, all metrics with bootstrap CIs, `seed`, `threshold_percentile` |
| `predictions/<fold>.npz` | observed and predicted series (`y_true`, `y_pred`) of each fold |
| `predictions/robustness_*.npz` | Monte Carlo replicas, mean, std and unperturbed baseline |
| `predictions/metrics_json/` | per-fold extended metrics |
| `predictions/final_report.json` | model selection summary and holdout metrics |
| `tests63/<hash_id>/` | content-addressed cache of each iteration: data bundle copy, windows, pretrained and fine-tuned `.keras` models, per-model predictions, OOF predictions, meta-learner |
| `doc/` | paper figures (PDF/EPS/PNG, 300 DPI) and LaTeX tables |

The `hash_id` covers every hyperparameter that affects training, so a changed
configuration never reuses stale weights. Model checkpoints and caches are
excluded from version control.

---

## 7. License

Released under the MIT License (see `LICENSE`).
