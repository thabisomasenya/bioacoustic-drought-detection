"""Input pipeline and noise augmentation.

Real deployment means plant clicks arrive mixed with microphone self-noise and
greenhouse sounds (fans, irrigation, people). The training pipeline simulates
this by mixing each training clip, on the fly, with a randomly chosen clip from
the *training* noise bank (Empty Pot + Greenhouse Noises) at a random
signal-to-noise ratio:

    gain   = sqrt( P_signal / (P_noise * 10^(SNR/10)) )
    mixed  = signal + gain * noise

Power is measured on band-passed clips, so the SNR is *in-band*, i.e. in the
20-100 kHz region the model actually sees. Because each epoch draws fresh noise,
fresh SNRs and fresh shifts, the network never sees the same input twice, which
regularises a model trained on only ~2,000 clips.

Additional label-preserving transforms:
* noise clips are circularly rolled by a random offset, so a noise burst can
  land anywhere in the window, not only on the plant peak;
* small circular time shifts (+/- 20 samples = 40 us) simulate peak-detector jitter;
* polarity inversion: pressure sign depends on the microphone, not the plant.

Randomness uses *stateless* TF ops seeded from a ``tf.data.Dataset.random``
stream, so a given config and seed produce the same augmented batches every run.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def gain_for_snr(p_signal, p_noise, snr_db):
    """Noise gain giving the target SNR. Works for NumPy and TF tensors."""
    return (p_signal / (p_noise * 10.0 ** (snr_db / 10.0))) ** 0.5


def mix_at_snr(signals: np.ndarray, noise_bank: np.ndarray, snr_db: float,
               seed: int = 0) -> np.ndarray:
    """Deterministic NumPy mixing at one fixed SNR (used for robustness testing)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(noise_bank), size=len(signals))
    shifts = rng.integers(0, signals.shape[1], size=len(signals))
    noise = np.stack([np.roll(noise_bank[i], s) for i, s in zip(idx, shifts)])
    p_s = np.mean(signals.astype(np.float64) ** 2, axis=1, keepdims=True)
    p_n = np.mean(noise.astype(np.float64) ** 2, axis=1, keepdims=True)
    g = gain_for_snr(p_s, np.maximum(p_n, 1e-12), snr_db)
    return (signals + g * noise).astype(np.float32)


def _peak_normalise(x):
    return x / tf.maximum(tf.reduce_max(tf.abs(x)), 1e-8)


def make_augmenter(noise_bank: np.ndarray | None, acfg: dict, per_clip_peak: bool):
    """Build the per-example augmentation function used inside ``tf.data``."""
    use_noise = noise_bank is not None and len(noise_bank) > 0
    bank = tf.constant(noise_bank, tf.float32) if use_noise else None
    n_bank = len(noise_bank) if use_noise else 0
    lo, hi = (float(v) for v in acfg["snr_db"])
    max_shift = int(acfg["max_shift"])
    p_noise = float(acfg["noise_prob"])
    p_flip = float(acfg["polarity_flip_prob"])

    def augment(x, y, seed_pair):
        # Derive independent stateless seeds for each random decision.
        seeds = tf.random.experimental.stateless_split(seed_pair, num=7)
        length = tf.shape(x)[0]

        if max_shift > 0:
            shift = tf.random.stateless_uniform([], seeds[0], -max_shift, max_shift + 1, tf.int32)
            x = tf.roll(x, shift, axis=0)

        flip = tf.random.stateless_uniform([], seeds[1]) < p_flip
        x = tf.where(flip, -x, x)

        if use_noise:
            idx = tf.random.stateless_uniform([], seeds[2], 0, n_bank, tf.int32)
            offset = tf.random.stateless_uniform([], seeds[3], 0, length, tf.int32)
            noise = tf.roll(tf.gather(bank, idx), offset, axis=0)
            noise_flip = tf.random.stateless_uniform([], seeds[6]) < 0.5
            noise = tf.where(noise_flip, -noise, noise)
            snr = tf.random.stateless_uniform([], seeds[4], lo, hi)
            p_s = tf.reduce_mean(tf.square(x))
            p_n = tf.maximum(tf.reduce_mean(tf.square(noise)), 1e-12)
            mixed = x + gain_for_snr(p_s, p_n, snr) * noise
            apply = tf.random.stateless_uniform([], seeds[5]) < p_noise
            x = tf.where(apply, mixed, x)

        if per_clip_peak:
            x = _peak_normalise(x)
        return x, y

    return augment


def make_dataset(X: np.ndarray, y: np.ndarray, cfg: dict, training: bool,
                 noise_bank: np.ndarray | None = None, seed: int = 0,
                 batch_size: int | None = None) -> tf.data.Dataset:
    """tf.data pipeline -> batches of (waveform[..., 1], label).

    Training: shuffle each epoch + on-the-fly augmentation (if enabled).
    Validation/test: no randomness, so evaluation is exactly repeatable.
    """
    batch_size = batch_size or cfg["training"]["batch_size"]
    per_clip_peak = cfg["preprocessing"]["scaling"] == "peak"
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))

    if training:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
        acfg = cfg["augmentation"]
        if acfg.get("enabled", False):
            seeds = tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True).batch(2)
            ds = tf.data.Dataset.zip((ds, seeds)).map(
                lambda xy, s: make_augmenter(noise_bank, acfg, per_clip_peak)(xy[0], xy[1], s),
                num_parallel_calls=AUTOTUNE, deterministic=True)
        elif per_clip_peak:
            ds = ds.map(lambda x, t: (_peak_normalise(x), t))
    elif per_clip_peak:
        ds = ds.map(lambda x, t: (_peak_normalise(x), t))

    ds = ds.map(lambda x, t: (tf.expand_dims(x, -1), t))
    return ds.batch(batch_size).prefetch(AUTOTUNE)
