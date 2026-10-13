"""Shared fixtures: a tiny synthetic dataset mirroring the real folder layout."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drought_acoustics.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SR, T = 500_000, 1001


def click(freq_hz: float, rng, amp: float = 0.3) -> np.ndarray:
    """Synthetic ultrasonic click: decaying tone burst centred at sample 500."""
    t = (np.arange(T) - 500) / SR
    env = np.exp(-np.abs(t) / 40e-6)
    return (amp * env * np.sin(2 * np.pi * freq_hz * t) + 0.01 * rng.standard_normal(T)).astype(np.float32)


@pytest.fixture
def cfg(tmp_path):
    c = load_config(base=ROOT / "configs" / "default.yaml")
    c["project"]["artifacts_dir"] = str(tmp_path / "artifacts")
    c["project"]["runs_dir"] = str(tmp_path / "runs")
    c["data"]["raw_dir"] = str(tmp_path / "PlantSounds")
    return c


@pytest.fixture
def tiny_dataset(cfg):
    """4 classes x 6 plants x 5 clips + 2 noise folders, written as real .wav files."""
    rng = np.random.default_rng(0)
    raw = Path(cfg["data"]["raw_dir"])
    freqs = {"Tomato Cut": 40e3, "Tobacco Cut": 45e3, "Tomato Dry": 70e3, "Tobacco Dry": 75e3}
    for folder, f in freqs.items():
        (raw / folder).mkdir(parents=True)
        for plant in range(6):
            for s in range(5):
                sf.write(raw / folder / f"id_{plant}_sound_{s}.wav", click(f, rng), SR, subtype="PCM_16")
    for folder in ["Empty Pot", "Greenhouse Noises"]:
        (raw / folder).mkdir(parents=True)
        for s in range(20):
            sf.write(raw / folder / f"id_{s % 2}_sound_{s}.wav",
                     (0.05 * rng.standard_normal(T)).astype(np.float32), SR, subtype="PCM_16")
    return raw
