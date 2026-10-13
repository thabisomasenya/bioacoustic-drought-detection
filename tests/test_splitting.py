import numpy as np
import pandas as pd

from drought_acoustics.splitting import (assert_no_group_leakage, grouped_cv, grouped_holdout,
                                         split_noise)


def fake_signals(n_plants=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_plants):
        label, species = p % 2, ["tomato", "tobacco"][(p // 2) % 2]
        for _ in range(rng.integers(1, 40)):
            rows.append({"group": f"f{label}{species}/{p}", "label": label, "species": species})
    return pd.DataFrame(rows)


def test_holdout_is_partition_without_leakage(cfg):
    df = fake_signals()
    s = grouped_holdout(df, cfg)
    allidx = np.concatenate(list(s.values()))
    assert len(allidx) == len(df) == len(np.unique(allidx))
    assert_no_group_leakage(df, s)
    for part in s.values():
        assert set(df["label"].iloc[part]) == {0, 1}


def test_holdout_is_deterministic(cfg):
    df = fake_signals()
    a, b = grouped_holdout(df, cfg), grouped_holdout(df, cfg)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_cv_test_folds_cover_every_plant_once(cfg):
    df = fake_signals()
    folds = grouped_cv(df, cfg)
    assert len(folds) == cfg["split"]["cv_folds"]
    test_groups = [set(df["group"].iloc[f["test"]]) for f in folds]
    assert set().union(*test_groups) == set(df["group"])
    assert sum(len(g) for g in test_groups) == df["group"].nunique()


def test_leakage_detector_fires():
    df = pd.DataFrame({"group": ["a", "a", "b"]})
    try:
        assert_no_group_leakage(df, {"train": np.array([0]), "test": np.array([1, 2])})
    except AssertionError:
        return
    raise AssertionError("leakage not detected")


def test_noise_split_is_disjoint(cfg):
    noise = pd.DataFrame({"noise_source": ["empty_pot"] * 40 + ["greenhouse"] * 60})
    s = split_noise(noise, cfg)
    assert not (set(s["train"]) & set(s["test"])) and not (set(s["val"]) & set(s["test"]))


def test_holdout_partitions_are_representative(cfg):
    """Each partition's class x species clip mix stays close to the full dataset's."""
    df = fake_signals(n_plants=120, seed=3)
    s = grouped_holdout(df, cfg)
    strata = df["label"].astype(str) + df["species"]
    overall = strata.value_counts(normalize=True)
    for part in s.values():
        got = strata.iloc[part].value_counts(normalize=True).reindex(overall.index, fill_value=0)
        assert (got - overall).abs().max() < 0.08
