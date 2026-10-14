"""Evaluation and testing.

Metric choice
-------------
* **Macro F1** and **precision**: the Deliverable 1 metrics. Macro F1 averages
  F1 over both classes, so the minority (cut) class counts equally.
* **ROC-AUC / PR-AUC**: threshold-free ranking quality.
* **Balanced accuracy, MCC**: robust single-number summaries under imbalance.
* **Recall (sensitivity) for drought**: in an early-warning system a missed
  drought event is the costly error, so this is reported separately.

Protocol
--------
1. The decision threshold is chosen on the *validation* set (max macro F1) and
   then frozen, because tuning it on test would leak.
2. Test metrics come with 95% CIs from a **plant-level (cluster) bootstrap**:
   plants, not clips, are resampled, because clips from one plant are correlated.
3. Results are broken down **per species** to expose any species shortcut.
4. **Plant-level** scores average a plant's clip probabilities, which is the
   decision a grower actually needs ("is *this plant* drought-stressed?").
5. **Robustness**: test clips are re-mixed with *held-out* noise at fixed SNRs
   (20 to -5 dB) to measure degradation under realistic interference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from .augmentation import make_dataset, mix_at_snr


def predict(model, X: np.ndarray, cfg: dict) -> np.ndarray:
    ds = make_dataset(X, np.zeros(len(X)), cfg, training=False, batch_size=256)
    return model.predict(ds, verbose=0).ravel()


def choose_threshold(y: np.ndarray, p: np.ndarray, mode: str = "val_f1") -> float:
    if mode != "val_f1":
        return 0.5
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, p >= t, average="macro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_drought": float(precision_score(y, pred, zero_division=0)),
        "recall_drought": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1_drought": float(f1_score(y, pred, zero_division=0)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Always predict the training majority class: the floor any model must beat."""
    p = np.full(len(y_test), float(np.round(np.mean(y_train))))
    m = compute_metrics(y_test, p, 0.5)
    m["roc_auc"] = 0.5
    return m


def bootstrap_ci(pred_df: pd.DataFrame, threshold: float, n_iter: int = 1000,
                 seed: int = 0, metrics=("macro_f1", "roc_auc", "balanced_accuracy")) -> dict:
    """95% CIs by resampling *plants* with replacement (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    groups = pred_df["group"].unique()
    by_group = {g: d for g, d in pred_df.groupby("group")}
    samples = {m: [] for m in metrics}
    for _ in range(n_iter):
        pick = rng.choice(groups, size=len(groups), replace=True)
        d = pd.concat([by_group[g] for g in pick])
        if d["label"].nunique() < 2:
            continue
        res = compute_metrics(d["label"].to_numpy(), d["prob"].to_numpy(), threshold)
        for m in metrics:
            samples[m].append(res[m])
    return {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for m, v in samples.items()}


def per_species(pred_df: pd.DataFrame, threshold: float) -> dict:
    return {sp: compute_metrics(d["label"].to_numpy(), d["prob"].to_numpy(), threshold)
            for sp, d in pred_df.groupby("species")}


def plant_level(pred_df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict]:
    """Aggregate clip probabilities to one score per plant (mean)."""
    plants = (pred_df.groupby("group")
              .agg(label=("label", "first"), species=("species", "first"),
                   prob=("prob", "mean"), n_clips=("prob", "size"))
              .reset_index())
    return plants, compute_metrics(plants["label"].to_numpy(), plants["prob"].to_numpy(), threshold)


def robustness(model, X_test: np.ndarray, y_test: np.ndarray, noise_bank: np.ndarray,
               noise_sources: np.ndarray, cfg: dict, threshold: float) -> pd.DataFrame:
    """Metrics on test clips re-mixed with held-out noise at fixed SNRs."""
    rows = []
    seed = cfg["project"]["seed"]
    p_clean = predict(model, X_test, cfg)
    clean = compute_metrics(y_test, p_clean, threshold)
    for source in ["all", *sorted(set(noise_sources))]:
        rows.append({"noise_source": source, "snr_db": "clean", **clean})
        bank = noise_bank if source == "all" else noise_bank[noise_sources == source]
        for snr in cfg["evaluation"]["snr_levels_db"]:
            X_mix = mix_at_snr(X_test, bank, snr, seed=seed + 1000 + int(snr * 10))
            rows.append({"noise_source": source, "snr_db": snr,
                         **compute_metrics(y_test, predict(model, X_mix, cfg), threshold)})
    return pd.DataFrame(rows)


def corrected_resampled_ttest(diffs: np.ndarray) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected paired t-test for k-fold CV score differences.

    Fold scores share most of their training data, so the ordinary paired t-test
    underestimates the variance and finds "significant" differences too easily.
    The correction inflates the variance by rho / (1 - rho) with rho = 1/k:

        t = mean(d) / sqrt((1/k + 1/(k-1)) * var(d)),   df = k - 1
    """
    from scipy import stats

    d = np.asarray(diffs, dtype=float)
    k = len(d)
    if k < 2:
        return float("nan"), float("nan")
    var = d.var(ddof=1)
    if var == 0:
        return float("inf") if d.mean() != 0 else 0.0, 0.0 if d.mean() != 0 else 1.0
    t = d.mean() / np.sqrt((1 / k + 1 / (k - 1)) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def paired_bootstrap_auc_diff(a: pd.DataFrame, b: pd.DataFrame, n_iter: int = 1000,
                              seed: int = 0) -> dict:
    """95% CI for AUC(a) - AUC(b) on the SAME clips, resampling plants (cluster bootstrap).

    ``a`` and ``b`` are out-of-fold prediction frames (path, group, label, prob) from two
    models evaluated on identical folds. Pairing on clips removes the between-plant
    variation the two models share, which makes this far more sensitive than
    comparing two separate confidence intervals.
    """
    m = a[["path", "group", "label", "prob"]].merge(b[["path", "prob"]], on="path",
                                                     suffixes=("_a", "_b"))
    rng = np.random.default_rng(seed)
    by_group = {g: d for g, d in m.groupby("group")}
    groups = np.array(list(by_group))
    point = roc_auc_score(m["label"], m["prob_a"]) - roc_auc_score(m["label"], m["prob_b"])
    diffs = []
    for _ in range(n_iter):
        d = pd.concat([by_group[g] for g in rng.choice(groups, size=len(groups), replace=True)])
        if d["label"].nunique() < 2:
            continue
        diffs.append(roc_auc_score(d["label"], d["prob_a"]) - roc_auc_score(d["label"], d["prob_b"]))
    diffs = np.array(diffs)
    return {"delta_auc": float(point), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "p_one_sided": float(np.mean(diffs <= 0)), "n_clips": int(len(m))}
