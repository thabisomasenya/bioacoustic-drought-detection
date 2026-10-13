"""Signal preprocessing for 500 kHz ultrasonic clips.

Order of operations (applied identically to plant clips and noise clips):

1. **DC removal**: subtracts each clip's mean so a microphone offset cannot act
   as a spurious feature.
2. **Butterworth band-pass (20-100 kHz)**: Deliverable 1's design. The Butterworth
   response is maximally flat in the pass-band, so in-band amplitudes are not
   distorted. It is applied forward and backward (``sosfiltfilt``), which gives
   *zero phase*: the transient peak stays at sample 500 instead of being delayed,
   which matters when the whole clip is only 2 ms. Second-order sections are used
   for numerical stability at these normalised cut-offs.
3. **Amplitude scaling**: a single global factor fitted on the *training split
   only* (no test-set leakage) brings inputs to unit RMS for stable optimisation
   while preserving relative loudness between clips, which is itself informative.

Because steps 1-2 are linear, filtering noise and signal separately and then
mixing them is mathematically identical to filtering the mixture. This lets the
augmentation stage work on pre-filtered arrays inside ``tf.data``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def design_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2
    if not 0 < low_hz < high_hz < nyq:
        raise ValueError(f"Invalid band {low_hz}-{high_hz} Hz for fs={fs}")
    return butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")


def filter_signals(X: np.ndarray, cfg: dict) -> np.ndarray:
    """DC removal + zero-phase band-pass. Shape preserved: (N, T)."""
    p = cfg["preprocessing"]
    X = X.astype(np.float32, copy=True)
    if p.get("remove_dc", True):
        X -= X.mean(axis=1, keepdims=True)
    bp = p.get("bandpass", {})
    if bp.get("enabled", True):
        sos = design_bandpass(bp["low_hz"], bp["high_hz"], cfg["data"]["sample_rate"], bp["order"])
        X = sosfiltfilt(sos, X, axis=1).astype(np.float32)
    return X


class AmplitudeScaler:
    """Train-fitted amplitude normalisation (the audio analogue of StandardScaler).

    ``global_rms``: one factor for all clips (keeps relative loudness).
    ``peak``:       per-clip peak normalisation (removes loudness entirely).
    ``none``:       identity.
    """

    def __init__(self, method: str = "global_rms"):
        if method not in {"global_rms", "peak", "none"}:
            raise ValueError(f"Unknown scaling method: {method}")
        self.method = method
        self.scale_: float = 1.0

    def fit(self, X_train: np.ndarray) -> "AmplitudeScaler":
        if self.method == "global_rms":
            self.scale_ = float(np.sqrt(np.mean(X_train.astype(np.float64) ** 2)))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.method == "global_rms":
            return (X / self.scale_).astype(np.float32)
        if self.method == "peak":
            peak = np.abs(X).max(axis=1, keepdims=True)
            return (X / np.maximum(peak, 1e-8)).astype(np.float32)
        return X.astype(np.float32)

    def to_dict(self) -> dict:
        return {"method": self.method, "scale": self.scale_}

    @classmethod
    def from_dict(cls, d: dict) -> "AmplitudeScaler":
        s = cls(d["method"])
        s.scale_ = d["scale"]
        return s
