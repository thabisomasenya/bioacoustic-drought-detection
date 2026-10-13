"""Leakage-safe data splitting.

Each plant contributes many clicks (up to 148). Clicks from the same plant share
its anatomy, microphone position and recording session, so a random clip-level
split would put near-duplicates of test clips in the training set and inflate
every metric. Splits are therefore made at the **plant level** (all clips of a
plant land in exactly one partition) using ``StratifiedGroupKFold``, stratified
on the joint *class x species* label so every partition keeps the class balance
and the tomato/tobacco mix of the full dataset.

Noise clips come from very few recording sessions (one Empty Pot instance and
four greenhouse instances), so a session-level split is impossible. They are
split at file level, stratified by source, so test-time noise is at least never
seen during training. This limitation is recorded in the split summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .utils import get_logger

log = get_logger(__name__)


def _strata(df: pd.DataFrame) -> np.ndarray:
    return (df["label"].astype(str) + "_" + df["species"]).to_numpy()


def split_score(sub: pd.DataFrame, held: np.ndarray, fraction: float,
                min_plants: int = 3) -> float:
    """How far a candidate held-out set is from being a faithful miniature of ``sub``.

    Sum of (i) L1 distance between the held-out and overall class x species clip
    proportions, (ii) deviation from the target size, and (iii) a penalty for
    every stratum represented by fewer than ``min_plants`` plants.
    """
    strata = pd.Series(_strata(sub))
    target = strata.value_counts(normalize=True)
    got = strata.iloc[held].value_counts(normalize=True).reindex(target.index, fill_value=0)
    plants = sub.iloc[held].assign(s=strata.iloc[held].to_numpy()).groupby("s")["group"].nunique()
    plants = plants.reindex(target.index, fill_value=0)
    short = int(np.clip(min_plants - plants, 0, None).sum())
    return float(np.abs(got - target).sum() + abs(len(held) / len(sub) - fraction) + 0.25 * short)


def _one_group_fold(df: pd.DataFrame, idx: np.ndarray, fraction: float, seed: int,
                    n_candidates: int = 40):
    """Split ``idx`` into (rest, held_out) with ~``fraction`` of the clips held out.

    Plants contribute between 1 and 148 clips, so a single ``StratifiedGroupKFold``
    draw can produce a held-out set whose class/species mix is badly skewed (it can
    stratify plant counts but not clip counts). We therefore generate candidate
    folds from ``n_candidates`` seeded shuffles and keep the one with the lowest
    ``split_score``. The search is deterministic for a given seed.
    """
    n_splits = max(2, int(round(1 / fraction)))
    sub = df.iloc[idx].reset_index(drop=True)
    best, best_score = None, np.inf
    for c in range(n_candidates):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + c)
        for rest, held in sgkf.split(sub, _strata(sub), groups=sub["group"]):
            score = split_score(sub, held, fraction)
            if score < best_score:
                best, best_score = (rest, held), score
    rest, held = best
    return idx[rest], idx[held]


def grouped_holdout(signals: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    """Plant-grouped, stratified train/val/test indices into ``signals``."""
    s = cfg["split"]
    seed = cfg["project"]["seed"]
    all_idx = np.arange(len(signals))
    trainval, test = _one_group_fold(signals, all_idx, s["test_size"], seed)
    val_frac = s["val_size"] / (1 - s["test_size"])
    train, val = _one_group_fold(signals, trainval, val_frac, seed + 1)
    splits = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    assert_no_group_leakage(signals, splits)
    return splits


def grouped_cv(signals: pd.DataFrame, cfg: dict) -> list[dict[str, np.ndarray]]:
    """K plant-grouped folds; inside each, a grouped validation set is carved out."""
    s = cfg["split"]
    seed = cfg["project"]["seed"]
    k = s["cv_folds"]
    # Same idea as the holdout: pick, among seeded candidates, the partition whose
    # folds are jointly the most representative.
    best, best_score = None, np.inf
    for c in range(40):
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + c)
        cand = list(sgkf.split(signals, _strata(signals), groups=signals["group"]))
        score = sum(split_score(signals, test, 1 / k) for _, test in cand)
        if score < best_score:
            best, best_score = cand, score
    folds = []
    for trainval, test in best:
        val_frac = s["val_size"] / (1 - 1 / s["cv_folds"])
        train, val = _one_group_fold(signals, trainval, val_frac, seed + 1)
        fold = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
        assert_no_group_leakage(signals, fold)
        folds.append(fold)
    return folds


def split_noise(noise: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    """File-level, source-stratified split of noise clips."""
    tr, va, te = cfg["split"]["noise_split"]
    seed = cfg["project"]["seed"]
    idx = np.arange(len(noise))
    train, rest = train_test_split(idx, train_size=tr, stratify=noise["noise_source"],
                                   random_state=seed)
    val, test = train_test_split(rest, train_size=va / (va + te),
                                 stratify=noise["noise_source"].iloc[rest], random_state=seed)
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def assert_no_group_leakage(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> None:
    groups = {k: set(df["group"].iloc[v]) for k, v in splits.items()}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = groups[a] & groups[b]
            if overlap:
                raise AssertionError(f"Plant leakage between {a} and {b}: {sorted(overlap)[:5]}")


def summarise_split(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> pd.DataFrame:
    """Clips and plants per partition x class x species (for the report)."""
    rows = []
    for name, idx in splits.items():
        part = df.iloc[idx]
        for (label, species), g in part.groupby(["label", "species"]):
            rows.append({"split": name, "label": int(label), "species": species,
                         "clips": len(g), "plants": g["group"].nunique()})
    return pd.DataFrame(rows)
