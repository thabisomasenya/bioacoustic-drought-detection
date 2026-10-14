import numpy as np
import pandas as pd
from scipy import stats

from drought_acoustics.evaluation import (bootstrap_ci, choose_threshold, compute_metrics,
                                          corrected_resampled_ttest, majority_baseline, plant_level)


def test_perfect_and_inverted_predictions():
    y = np.array([0, 0, 1, 1])
    good = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    bad = compute_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]), 0.5)
    assert good["macro_f1"] == 1.0 and good["roc_auc"] == 1.0 and good["mcc"] == 1.0
    assert bad["roc_auc"] == 0.0 and bad["tp"] == 0


def test_threshold_is_chosen_from_scores_not_fixed():
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.linspace(0.05, 0.25, 50), np.linspace(0.3, 0.5, 50)]  # all below 0.5
    t = choose_threshold(y, p)
    assert 0.25 <= t <= 0.30
    assert compute_metrics(y, p, t)["macro_f1"] == 1.0


def test_majority_baseline_is_chance_level():
    m = majority_baseline(np.array([1, 1, 0]), np.array([0, 1, 1, 1]))
    assert m["roc_auc"] == 0.5 and m["recall_drought"] == 1.0 and m["specificity"] == 0.0


def test_plant_level_averages_clips_per_plant():
    df = pd.DataFrame({"group": ["a", "a", "b"], "label": [1, 1, 0], "species": ["t"] * 3,
                       "prob": [0.2, 0.8, 0.1]})
    plants, m = plant_level(df, 0.5)
    assert plants.set_index("group").loc["a", "prob"] == 0.5
    assert m["n"] == 2


def test_cluster_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    rows = [{"group": f"g{g}", "label": g % 2, "species": "t",
             "prob": float(np.clip(0.3 + 0.4 * (g % 2) + rng.normal(0, 0.2), 0, 1))}
            for g in range(40) for _ in range(5)]
    df = pd.DataFrame(rows)
    point = compute_metrics(df["label"].to_numpy(), df["prob"].to_numpy(), 0.5)["roc_auc"]
    lo, hi = bootstrap_ci(df, 0.5, n_iter=200, seed=1)["roc_auc"]
    assert lo <= point <= hi


def test_corrected_ttest_is_more_conservative_than_plain_ttest():
    d = np.array([0.035, 0.049, 0.009, 0.044, -0.018])
    t_c, p_c = corrected_resampled_ttest(d)
    t_plain, p_plain = stats.ttest_1samp(d, 0.0)
    assert abs(t_c) < abs(t_plain) and p_c > p_plain


def test_paired_bootstrap_detects_a_better_model():
    from drought_acoustics.evaluation import paired_bootstrap_auc_diff
    rng = np.random.default_rng(0)
    rows = [{"path": f"g{g}_{i}", "group": f"g{g}", "label": g % 2} for g in range(60) for i in range(4)]
    a = pd.DataFrame(rows)
    b = a.copy()
    a["prob"] = np.clip(0.25 + 0.5 * a["label"] + rng.normal(0, 0.15, len(a)), 0, 1)   # strong
    b["prob"] = np.clip(0.45 + 0.1 * b["label"] + rng.normal(0, 0.15, len(b)), 0, 1)   # weak
    res = paired_bootstrap_auc_diff(a, b, n_iter=200, seed=0)
    assert res["ci_low"] > 0 and res["delta_auc"] > 0
