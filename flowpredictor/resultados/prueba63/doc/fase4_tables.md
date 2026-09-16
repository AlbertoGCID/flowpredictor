## Tabla 1 auditada — CV expansiva 2017–2023 · 48 h · seed 92 · p90

| Config | n | NSE (media±std) | KGE (media±std) | HitRatio (media±std) | FARate | FARatio | F1 | Signed Lag (d) | Absolute Lag (d) | Wilcoxon p NSE vs M1 | Wilcoxon p Hit vs M1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 7 | 0.404 ± 0.175 | 0.490 ± 0.099 | 0.132 ± 0.096 | 0.015 | 0.320 | 0.498 | 1.032 | 1.679 | ref. | ref. |
| M2 | 7 | 0.412 ± 0.196 | 0.593 ± 0.110 | 0.338 ± 0.171 | 0.051 | 0.486 | 0.537 | 0.737 | 1.368 | 0.9375 (Δ +0.009) | ≈0.0277 (Δ +0.206) |
| M5 | 7 | -0.028 ± 0.523 | 0.304 ± 0.263 | 0.444 ± 0.151 | 0.109 | 0.567 | 0.546 | 0.378 | 1.288 | 0.0156 (Δ -0.432) | 0.0156 (Δ +0.311) |
| heuristic_ensemble | 7 | 0.082 ± 0.396 | 0.304 ± 0.178 | 0.426 ± 0.170 | 0.110 | 0.575 | 0.545 | 0.556 | 1.248 | 0.0156 (Δ -0.322) | 0.0156 (Δ +0.294) |
| quantile_regression | 7 | 0.391 ± 0.082 | 0.435 ± 0.079 | 0.044 ± 0.043 | 0.023 | 0.490 | 0.299 | 0.960 | 1.631 | 0.9375 (Δ -0.012) | ≈0.0431 (Δ -0.088) |
| gradient_boosting_quantile | 7 | 0.458 ± 0.089 | 0.441 ± 0.079 | 0.039 ± 0.041 | 0.017 | 0.221 | 0.380 | 0.564 | 1.329 | 0.3750 (Δ +0.054) | ≈0.0277 (Δ -0.093) |

Wilcoxon de rangos con signo pareado por año de test (7 folds); Δ = media entre folds de (config − M1). Sin diferencias nulas, con n=7 el p exacto mínimo es 0.0156. «≈»: algún fold con diferencia exactamente cero; scipy recurre a la aproximación normal, poco fiable con n=7.

## Tabla 2 — Baselines τ=0.50 vs τ=0.90 · 48 h · seed 92 · p90

| Baseline | τ | Estrategia | n | NSE | HitRatio | Recall | FARate |
|---|---|---|---|---|---|---|---|
| Quantile Regression | 0.50 | expanding | 7 | 0.391 ± 0.082 | 0.044 ± 0.043 | 0.225 ± 0.132 | 0.023 ± 0.017 |
| Quantile Regression | 0.90 | expanding | 7 | -0.593 ± 0.282 | 0.494 ± 0.233 | 0.913 ± 0.066 | 0.206 ± 0.102 |
| Quantile Regression | 0.50 | holdout_final | 1 | 0.379 | 0.000 | 0.179 | 0.015 |
| Quantile Regression | 0.90 | holdout_final | 1 | -0.699 | 0.357 | 0.893 | 0.190 |
| Gradient Boosting | 0.50 | expanding | 7 | 0.458 ± 0.089 | 0.039 ± 0.041 | 0.295 ± 0.205 | 0.017 ± 0.015 |
| Gradient Boosting | 0.90 | expanding | 7 | 0.241 ± 0.177 | 0.414 ± 0.194 | 0.791 ± 0.107 | 0.104 ± 0.059 |
| Gradient Boosting | 0.50 | holdout_final | 1 | 0.424 | 0.000 | 0.214 | 0.002 |
| Gradient Boosting | 0.90 | holdout_final | 1 | 0.020 | 0.321 | 0.750 | 0.127 |

## Tabla 3 — Sensibilidad de umbral p90 vs p95 · 48 h · seed 92

| Config | Estrategia | Umbral | n | HitRatio | FARate | Recall | N eventos (total) |
|---|---|---|---|---|---|---|---|
| M1 | expanding | p90 | 7 | 0.132 ± 0.096 | 0.015 ± 0.010 | 0.437 ± 0.231 | 251 |
| M1 | expanding | p95 | 7 | 0.098 ± 0.050 | 0.007 ± 0.007 | 0.241 ± 0.187 | 114 |
| M1 | holdout_final | p90 | 1 | 0.036 | 0.002 | 0.250 | 28 |
| M1 | holdout_final | p95 | 1 | 0.000 | 0.000 | 0.154 | 13 |
| M2 | expanding | p90 | 7 | 0.338 ± 0.171 | 0.051 ± 0.034 | 0.591 ± 0.232 | 251 |
| M2 | expanding | p95 | 7 | 0.258 ± 0.116 | 0.029 ± 0.025 | 0.406 ± 0.175 | 114 |
| M2 | holdout_final | p90 | 1 | 0.143 | 0.040 | 0.536 | 28 |
| M2 | holdout_final | p95 | 1 | 0.077 | 0.004 | 0.385 | 13 |
| M5 | expanding | p90 | 7 | 0.444 ± 0.151 | 0.109 ± 0.063 | 0.770 ± 0.050 | 251 |
| M5 | expanding | p95 | 7 | 0.296 ± 0.188 | 0.051 ± 0.030 | 0.632 ± 0.148 | 114 |
| M5 | holdout_final | p90 | 1 | 0.321 | 0.096 | 0.750 | 28 |
| M5 | holdout_final | p95 | 1 | 0.154 | 0.030 | 0.538 | 13 |

## Tabla 4 — Estabilidad por semillas (92, 42, 123) · p90 · 48 h

Por semilla se toma la media sobre los 7 folds de CV expansiva; la tabla resume esas medias entre semillas. El holdout (1 valor por semilla) va aparte.

Una semilla solo cuenta si tiene el conjunto COMPLETO de folds (7 expanding / 1 holdout); las parciales se listan aparte y no entran en los estadísticos.

| Config | Métrica | Estrategia | Semillas completas | Media | Min | Max | Std | Incompletas (folds) |
|---|---|---|---|---|---|---|---|---|
| M2 | NSE | expanding | [42, 92, 123] | 0.339 | 0.294 | 0.412 | 0.064 | — |
| M2 | NSE | holdout_final | [42, 92, 123] | 0.441 | 0.396 | 0.482 | 0.043 | — |
| M2 | HitRatio | expanding | [42, 92, 123] | 0.319 | 0.290 | 0.338 | 0.025 | — |
| M2 | HitRatio | holdout_final | [42, 92, 123] | 0.167 | 0.143 | 0.214 | 0.041 | — |
| M5 | NSE | expanding | [42, 92, 123] | -0.013 | -0.048 | 0.036 | 0.044 | — |
| M5 | NSE | holdout_final | [42, 92, 123] | 0.126 | 0.015 | 0.212 | 0.101 | — |
| M5 | HitRatio | expanding | [42, 92, 123] | 0.455 | 0.444 | 0.470 | 0.014 | — |
| M5 | HitRatio | holdout_final | [42, 92, 123] | 0.357 | 0.321 | 0.429 | 0.062 | — |

## Tabla 5 — Selección de M* recalculada · CV pura (expanding + LOYO, sin holdout) · seed 92 · p90

| Config | n folds | NSE | FARate | ΔHitRatio | α ≤ 0.10 | α ≤ 0.12 |
|---|---|---|---|---|---|---|
| M1 | 17 | 0.3998 | 0.0122 | -0.0766 | -0.0766 | -0.0766 |
| M2 | 17 | 0.4731 | 0.0478 | 0.1430 | 0.1430 | 0.1430 |
| M3 | 17 | 0.3932 | 0.0124 | -0.0766 | -0.0766 | -0.0766 |
| M4 | 17 | 0.4117 | 0.0574 | 0.1269 | 0.1269 | 0.1269 |
| M5 | 17 | 0.0835 | 0.1026 | 0.2925 | excluida | 0.2925 |

**M\* con α ≤ 0.10: M2**

**M\* con α ≤ 0.12: M5**