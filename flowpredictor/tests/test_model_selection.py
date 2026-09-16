"""Tests of the M1-M5 comparison and selection on results_consolidated.csv (combined expanding + LOYO evidence)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from resultados.prueba63.src.evaluation.model_selection import (
    BASELINE_CONFIGS,
    aggregate_config_metrics,
    build_final_report,
    compute_delta_hit_ratio,
    lexicographic_constrained_score,
    paired_fold_tests,
    run_model_selection,
    select_best_model,
    weighted_normalized_score,
)


def _row(config, test_year, strategy, hit_ratio, peak_timing, nse, farate=0.0):
    return {
        "fold": f"{config}_{strategy}_{test_year}", "test_year": test_year, "hash": "h",
        "config": config, "model": "seq2seq_base", "strategy": strategy,
        "HitRatio": hit_ratio, "PeakTiming_mean_absolute_lag": peak_timing, "NSE": nse,
        "FARate": farate,
    }


def _synthetic_df() -> pd.DataFrame:
    """2 expanding + 2 LOYO folds.

    M5 always wins on HitRatio/NSE and has a lower PeakTimingError than M1-M4,
    so it must win unambiguously under weighted_normalized_score. M5 has
    FARate=0.15 (> default alpha=0.10) while M1-M4 have FARate=0.05, so under
    the default lexicographic selection M5 is excluded despite winning on the
    other signals, and M4 wins (largest delta HitRatio among M1-M4). Baselines
    are fixed (HitRatio=0.5), so delta HitRatio(MX) = HitRatio(MX) - 0.5.
    """
    rows = []
    for strategy, years in (("expanding", [2017, 2018]), ("loyo", [2014, 2015])):
        for year in years:
            for b in ("quantile_regression", "gradient_boosting_quantile", "heuristic_ensemble"):
                rows.append(_row(b, year, strategy, hit_ratio=0.5, peak_timing=2.0, nse=0.5))
            for i, cfg in enumerate(["M1", "M2", "M3", "M4"], start=1):
                rows.append(_row(cfg, year, strategy, hit_ratio=0.5 + 0.02 * i, peak_timing=2.0 - 0.05 * i,
                                  nse=0.5 + 0.02 * i, farate=0.05))
            rows.append(_row("M5", year, strategy, hit_ratio=0.9, peak_timing=0.5, nse=0.9, farate=0.15))
    return pd.DataFrame(rows)


def test_compute_delta_hit_ratio_uses_mean_of_three_baselines_same_fold():
    df = _synthetic_df()
    out = compute_delta_hit_ratio(df)
    m5_2017 = out[(out["config"] == "M5") & (out["test_year"] == 2017) & (out["strategy"] == "expanding")]
    assert len(m5_2017) == 1
    # baselines fixed at 0.5 -> BaselineHitRatio=0.5 -> DeltaHitRatio = 0.9 - 0.5
    assert m5_2017["DeltaHitRatio"].iloc[0] == pytest.approx(0.4)


def test_compute_delta_hit_ratio_never_mixes_folds_across_years_or_strategies():
    """An 'expanding' fold must not use the 'loyo' baseline of the same year, nor
    vice versa: the join key is (test_year, strategy), not only test_year."""
    df = _synthetic_df()
    # Checked indirectly: every row must get exactly the baseline of ITS OWN
    # (test_year, strategy), not a global average.
    out = compute_delta_hit_ratio(df)
    for _, row in out.iterrows():
        expected_baseline = df[
            (df["config"].isin(["quantile_regression", "gradient_boosting_quantile", "heuristic_ensemble"]))
            & (df["test_year"] == row["test_year"])
            & (df["strategy"] == row["strategy"])
        ]["HitRatio"].mean()
        assert row["DeltaHitRatio"] == pytest.approx(row["HitRatio"] - expected_baseline)


def test_aggregate_config_metrics_reports_expanding_loyo_and_combined_separately():
    df = _synthetic_df()
    agg = aggregate_config_metrics(df)
    assert set(agg.keys()) == {"M1", "M2", "M3", "M4", "M5"}
    for cfg in agg:
        assert set(agg[cfg].keys()) == {"expanding", "loyo", "combined"}
        assert agg[cfg]["expanding"]["n_folds"] == 2
        assert agg[cfg]["loyo"]["n_folds"] == 2
        assert agg[cfg]["combined"]["n_folds"] == 4  # combined evidence = both strategies pooled

    # M5 has the highest mean NSE of the 5 configurations in both strategies.
    nse_means = {cfg: agg[cfg]["combined"]["NSE_mean"] for cfg in agg}
    assert max(nse_means, key=nse_means.get) == "M5"


def test_weighted_normalized_score_favors_higher_hitratio_nse_and_lower_timing():
    df = _synthetic_df()
    agg = aggregate_config_metrics(df)
    scores = weighted_normalized_score(agg)
    assert set(scores.keys()) == {"M1", "M2", "M3", "M4", "M5"}
    assert max(scores, key=scores.get) == "M5"
    # The worst score must be M1 (lowest HitRatio/NSE, highest PeakTimingError among the ablation configs).
    assert min(scores, key=scores.get) == "M1"


def test_weighted_normalized_score_normalizes_min_max_into_zero_one_range():
    df = _synthetic_df()
    agg = aggregate_config_metrics(df)
    scores = weighted_normalized_score(agg)
    for s in scores.values():
        assert -1e-9 <= s <= 1 + 1e-9


def test_select_best_model_does_not_compute_any_metric_itself():
    """The selector delegates all scoring to decision_fn: with a fixed decision_fn
    that ignores the real metrics, the result must follow that injection, not
    hard-coded logic."""
    df = _synthetic_df()
    agg = aggregate_config_metrics(df)

    def _always_pick_m2(_configs_metrics):
        return {"M1": 0.0, "M2": 1.0, "M3": 0.0, "M4": 0.0, "M5": 0.0}

    result = select_best_model(agg, decision_fn=_always_pick_m2)
    assert result["selected"] == "M2"
    assert result["decision_fn"] == "_always_pick_m2"


def test_select_best_model_default_decision_fn_matches_direct_call():
    """The default decision_fn is lexicographic_constrained_score: calling
    select_best_model() without it must match calling that function directly."""
    df = _synthetic_df()
    agg = aggregate_config_metrics(df)
    result = select_best_model(agg)
    assert result["selected"] == "M4"
    assert result["scores"] == lexicographic_constrained_score(agg)


def test_run_model_selection_end_to_end_from_csv_and_persists_json(tmp_path):
    df = _synthetic_df()
    csv_path = tmp_path / "results_consolidated.csv"
    df.to_csv(csv_path, index=False)
    out_json = tmp_path / "model_selection.json"

    result = run_model_selection(str(csv_path), out_json_path=str(out_json))

    assert result["selected"] == "M4"
    assert out_json.exists()
    with open(out_json, "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["selected"] == "M4"
    assert persisted["metrics_by_config"]["M5"]["expanding"]["n_folds"] == 2
    assert persisted["metrics_by_config"]["M5"]["loyo"]["n_folds"] == 2


def test_lexicographic_constrained_score_excludes_configs_above_farate_max():
    """M5 has the best delta HitRatio/NSE, but FARate=0.15 > alpha=0.10: it must be
    excluded (-inf), and M4 must win (largest delta HitRatio among M1-M4)."""
    df = _synthetic_df()
    agg = aggregate_config_metrics(df)
    scores = lexicographic_constrained_score(agg)
    assert scores["M5"] == float("-inf")
    assert max(scores, key=scores.get) == "M4"


def test_lexicographic_constrained_score_excludes_configs_with_nonpositive_nse():
    """A configuration with NSE <= 0 must be excluded, even with the best delta HitRatio."""
    agg = {
        "A": {"combined": {"DeltaHitRatio_mean": 0.9, "NSE_mean": -0.1, "FARate_mean": 0.01}},
        "B": {"combined": {"DeltaHitRatio_mean": 0.2, "NSE_mean": 0.3, "FARate_mean": 0.02}},
    }
    scores = lexicographic_constrained_score(agg)
    assert scores["A"] == float("-inf")
    assert scores["B"] == pytest.approx(0.2)
    assert max(scores, key=scores.get) == "B"


def test_lexicographic_constrained_score_raises_if_no_config_qualifies():
    agg = {
        "A": {"combined": {"DeltaHitRatio_mean": 0.5, "NSE_mean": -0.1, "FARate_mean": 0.01}},
        "B": {"combined": {"DeltaHitRatio_mean": 0.2, "NSE_mean": 0.3, "FARate_mean": 0.50}},
    }
    with pytest.raises(RuntimeError, match="Ninguna config cumple"):
        lexicographic_constrained_score(agg)


def test_lexicographic_constrained_score_alpha_is_configurable():
    """With a looser alpha, M5 (FARate=0.15) satisfies the constraint and wins,
    confirming that alpha is a real parameter, not a hard-coded value."""
    df = _synthetic_df()
    agg = aggregate_config_metrics(df)
    scores = lexicographic_constrained_score(agg, farate_max=0.20)
    assert scores["M5"] != float("-inf")
    assert max(scores, key=scores.get) == "M5"


def test_build_final_report_attaches_holdout_metrics_of_selected_config(tmp_path):
    """build_final_report joins M* with the strategy='holdout_final' row of THAT
    SAME configuration, not another one."""
    df = _synthetic_df()
    holdout_row = _row("M4", 2025, "holdout_final", hit_ratio=0.7, peak_timing=1.0, nse=0.6, farate=0.03)
    df = pd.concat([df, pd.DataFrame([holdout_row])], ignore_index=True)
    csv_path = tmp_path / "results_consolidated.csv"
    df.to_csv(csv_path, index=False)

    report = build_final_report(str(csv_path))

    assert report["selected"] == "M4"
    assert report["holdout_metrics"] is not None
    assert report["holdout_metrics"]["strategy"] == "holdout_final"
    assert report["holdout_metrics"]["config"] == "M4"
    assert report["holdout_metrics"]["NSE"] == pytest.approx(0.6)


def test_build_final_report_holdout_metrics_none_if_not_yet_run(tmp_path):
    """If the holdout_final row of the selected M* does not exist yet, the report
    shows None instead of raising."""
    df = _synthetic_df()
    csv_path = tmp_path / "results_consolidated.csv"
    df.to_csv(csv_path, index=False)

    report = build_final_report(str(csv_path))

    assert report["selected"] == "M4"
    assert report["holdout_metrics"] is None


def test_build_final_report_persists_json(tmp_path):
    df = _synthetic_df()
    holdout_row = _row("M4", 2025, "holdout_final", hit_ratio=0.7, peak_timing=1.0, nse=0.6, farate=0.03)
    df = pd.concat([df, pd.DataFrame([holdout_row])], ignore_index=True)
    csv_path = tmp_path / "results_consolidated.csv"
    df.to_csv(csv_path, index=False)
    out_json = tmp_path / "final_report.json"

    build_final_report(str(csv_path), out_json_path=str(out_json))

    assert out_json.exists()
    with open(out_json, "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["selected"] == "M4"
    assert persisted["holdout_metrics"]["config"] == "M4"


# --------------------------------------------------------------------------- #
# The holdout must never enter the selection evidence
# --------------------------------------------------------------------------- #

def _df_with_holdout():
    """CV (2 expanding folds) + 1 holdout_final row per configuration, with a
    deliberately anomalous holdout so that its inclusion shows in the mean."""
    rows = []
    for cfg, nse_cv, hr_cv in (("M1", 0.40, 0.10), ("M2", 0.50, 0.30)):
        for year in (2017, 2018):
            rows.append({"config": cfg, "test_year": year, "strategy": "expanding",
                         "horizon": "48h", "NSE": nse_cv, "HitRatio": hr_cv,
                         "FARate": 0.02, "PeakTiming_mean_absolute_lag": 1.0})
        rows.append({"config": cfg, "test_year": 2025, "strategy": "holdout_final",
                     "horizon": "48h", "NSE": 10.0, "HitRatio": 10.0,
                     "FARate": 0.02, "PeakTiming_mean_absolute_lag": 1.0})
    for b in BASELINE_CONFIGS:
        for year, strat in ((2017, "expanding"), (2018, "expanding"), (2025, "holdout_final")):
            rows.append({"config": b, "test_year": year, "strategy": strat,
                         "horizon": "48h", "NSE": 0.3, "HitRatio": 0.05,
                         "FARate": 0.01, "PeakTiming_mean_absolute_lag": 1.5})
    return pd.DataFrame(rows)


def test_aggregate_excludes_holdout_by_default():
    agg = aggregate_config_metrics(_df_with_holdout(), horizon="48h")
    # 2 CV folds, not 3: the holdout_final row is excluded.
    assert agg["M2"]["combined"]["n_folds"] == 2
    assert agg["M2"]["combined"]["NSE_mean"] == pytest.approx(0.50)


def test_aggregate_can_opt_back_into_holdout():
    agg = aggregate_config_metrics(_df_with_holdout(), horizon="48h", exclude_holdout=False)
    assert agg["M2"]["combined"]["n_folds"] == 3
    assert agg["M2"]["combined"]["NSE_mean"] > 1.0  # contaminated by the holdout


def test_paired_fold_tests_reports_diff_and_pvalue():
    rows = []
    for i, year in enumerate(range(2017, 2024)):
        rows.append({"config": "M1", "test_year": year, "strategy": "expanding",
                     "horizon": "48h", "NSE": 0.30 + 0.01 * i, "HitRatio": 0.10})
        rows.append({"config": "M2", "test_year": year, "strategy": "expanding",
                     "horizon": "48h", "NSE": 0.45 + 0.01 * i, "HitRatio": 0.30})
    out = paired_fold_tests(pd.DataFrame(rows), reference="M1", configs=["M2"])

    assert out["M2"]["n_folds"] == 7
    assert out["M2"]["NSE"]["mean_diff"] == pytest.approx(0.15)
    assert out["M2"]["NSE"]["wilcoxon_p"] is not None
    assert out["M2"]["NSE"]["wilcoxon_p"] < 0.05
    assert out["M2"]["NSE"]["n_zero_diffs"] == 0
    assert out["M2"]["NSE"]["p_exact"] is True


def test_paired_fold_tests_returns_none_when_test_undefined():
    """All-zero differences: Wilcoxon is undefined and must be reported as None
    instead of raising."""
    rows = []
    for year in range(2017, 2024):
        for cfg in ("M1", "M2"):
            rows.append({"config": cfg, "test_year": year, "strategy": "expanding",
                         "horizon": "48h", "NSE": 0.4, "HitRatio": 0.2})
    out = paired_fold_tests(pd.DataFrame(rows), reference="M1", configs=["M2"])
    assert out["M2"]["NSE"]["wilcoxon_p"] is None
    assert out["M2"]["NSE"]["mean_diff"] == pytest.approx(0.0)
    assert out["M2"]["NSE"]["n_zero_diffs"] == 7
    assert out["M2"]["NSE"]["p_exact"] is False


# --------------------------------------------------------------------------- #
# Isolation of sensitivity variants (seed / p95 threshold)
# --------------------------------------------------------------------------- #

def test_filter_published_variant_drops_sensitivity_rows():
    from resultados.prueba63.src.evaluation.model_selection import filter_published_variant

    df = pd.DataFrame([
        {"config": "M2", "seed": 92, "threshold_percentile": 90, "NSE": 0.5},
        {"config": "M2", "seed": 92, "threshold_percentile": 95, "NSE": 0.1},
        {"config": "M2", "seed": 42, "threshold_percentile": 90, "NSE": 0.2},
        {"config": "M2", "seed": None, "threshold_percentile": None, "NSE": 0.6},
    ])
    assert sorted(filter_published_variant(df)["NSE"]) == [0.5, 0.6]


def test_aggregate_ignores_p95_rows_sharing_config_strategy_and_horizon():
    """p95 rows share config/strategy/horizon with the published ones: without the
    variant filter they would contaminate the selection of M*."""
    base = _df_with_holdout()
    base["seed"] = 92
    base["threshold_percentile"] = 90
    p95 = base.copy()
    p95["threshold_percentile"] = 95
    p95["NSE"] = -5.0

    agg = aggregate_config_metrics(pd.concat([base, p95], ignore_index=True), horizon="48h")
    assert agg["M2"]["combined"]["n_folds"] == 2
    assert agg["M2"]["combined"]["NSE_mean"] == pytest.approx(0.50)
