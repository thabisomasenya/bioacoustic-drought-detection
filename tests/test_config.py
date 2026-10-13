from drought_acoustics.config import config_hash, deep_merge, load_config

from conftest import ROOT


def test_deep_merge_keeps_unrelated_keys():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 10}})
    assert out == {"a": {"b": 10, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1  # input not mutated


def test_experiment_and_cli_overrides():
    cfg = load_config(ROOT / "configs/experiments/cnn1d_noaug.yaml",
                      ["training.epochs=7", "data.species=[tomato]"],
                      base=ROOT / "configs/default.yaml")
    assert cfg["augmentation"]["enabled"] is False
    assert cfg["training"]["epochs"] == 7
    assert cfg["data"]["species"] == ["tomato"]
    assert cfg["augmentation"]["snr_db"] == [0.0, 20.0]  # inherited from default


def test_config_hash_is_stable_and_sensitive():
    a = {"x": 1, "y": [1, 2]}
    assert config_hash(a) == config_hash({"y": [1, 2], "x": 1})
    assert config_hash(a) != config_hash({"x": 2, "y": [1, 2]})
