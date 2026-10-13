"""Model architectures.

``cnn1d`` (proposed model)
    A 1D CNN that learns directly from the 1,001-sample raw waveform. The block
    structure (paired Conv1D layers with kernel 9 and 32/64/128 filters, then 4x
    max-pooling and dropout, then a 128-unit dense head) follows the reference
    implementation released with Khait et al. (2023). This project adds optional
    batch normalisation, explicit layer naming and config-driven hyperparameters.

    Rationale: at 500 kHz, one sample is 2 us. A kernel of 9 samples spans 18 us,
    roughly one cycle of a 55 kHz tone (the spectral centre of these emissions).
    So the first layer learns band-pass-like filters matched to the click, and
    pooling builds a time-frequency hierarchy *learned from data* instead of one
    fixed by an STFT window.

``cnn2d_mel`` / ``lstm_mel`` (Deliverable 1 baselines)
    Both consume a log-Mel spectrogram computed *inside the graph* by
    ``LogMelSpectrogram``, so they receive exactly the same augmented waveforms
    as the 1D CNN and the comparison is fair. The Mel filterbank comes from
    Librosa. With n_fft=256 (0.51 ms, 25% of the clip) the spectrogram has only
    47 frames, and Deliverable 4 measures what that time-frequency trade-off costs.
"""

from __future__ import annotations

import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="drought_acoustics")
class LogMelSpectrogram(layers.Layer):
    """Waveform (B, T, 1) -> log-Mel spectrogram (B, frames, n_mels)."""

    def __init__(self, sample_rate=500000, n_fft=256, hop_length=16, n_mels=32,
                 fmin=20000.0, fmax=100000.0, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate, self.n_fft, self.hop_length = sample_rate, n_fft, hop_length
        self.n_mels, self.fmin, self.fmax = n_mels, fmin, fmax
        mel = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
                                  fmin=fmin, fmax=fmax)            # (n_mels, n_fft//2+1)
        self.mel_matrix = tf.constant(mel.T.astype(np.float32))    # (bins, n_mels)

    def call(self, x):
        x = tf.squeeze(x, axis=-1)
        stft = tf.signal.stft(x, frame_length=self.n_fft, frame_step=self.hop_length,
                              fft_length=self.n_fft, window_fn=tf.signal.hann_window)
        power = tf.square(tf.abs(stft))
        mel = tf.tensordot(power, self.mel_matrix, axes=1)
        return tf.math.log(mel + 1e-6)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length,
                   n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        return cfg


def build_cnn1d(input_len: int, p: dict) -> keras.Model:
    inp = keras.Input(shape=(input_len, 1), name="waveform")
    x = inp
    for b, f in enumerate(p["filters"], start=1):
        for c in (1, 2):
            x = layers.Conv1D(f, p["kernel_size"], padding="same",
                              use_bias=not p.get("batch_norm"), name=f"block{b}_conv{c}")(x)
            if p.get("batch_norm"):
                x = layers.BatchNormalization(name=f"block{b}_bn{c}")(x)
            x = layers.Activation("relu", name=f"block{b}_relu{c}")(x)
        x = layers.MaxPooling1D(p["pool_size"], name=f"block{b}_pool")(x)
        x = layers.Dropout(p["dropout"], name=f"block{b}_drop")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(p["dense_units"], activation="relu", name="dense")(x)
    x = layers.Dropout(p["dropout"], name="dense_drop")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_drought")(x)
    return keras.Model(inp, out, name="cnn1d")


def _mel_frontend(inp, sample_rate: int, mel: dict):
    x = LogMelSpectrogram(sample_rate=sample_rate, **mel, name="log_mel")(inp)
    return layers.BatchNormalization(name="mel_norm")(x)       # (B, frames, n_mels)


def build_cnn2d_mel(input_len: int, sample_rate: int, mel: dict, p: dict) -> keras.Model:
    inp = keras.Input(shape=(input_len, 1), name="waveform")
    x = layers.Reshape((-1, mel["n_mels"], 1), name="as_image")(_mel_frontend(inp, sample_rate, mel))
    for b, f in enumerate(p["filters"], start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"block{b}_conv")(x)
        x = layers.BatchNormalization(name=f"block{b}_bn")(x)
        x = layers.Activation("relu", name=f"block{b}_relu")(x)
        x = layers.MaxPooling2D(2, name=f"block{b}_pool")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(p["dense_units"], activation="relu", name="dense")(x)
    x = layers.Dropout(p["dropout"], name="dense_drop")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_drought")(x)
    return keras.Model(inp, out, name="cnn2d_mel")


def build_lstm_mel(input_len: int, sample_rate: int, mel: dict, p: dict) -> keras.Model:
    inp = keras.Input(shape=(input_len, 1), name="waveform")
    x = _mel_frontend(inp, sample_rate, mel)
    x = layers.LSTM(p["units"], name="lstm")(x)
    x = layers.Dropout(p["dropout"], name="lstm_drop")(x)
    x = layers.Dense(p["dense_units"], activation="relu", name="dense")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_drought")(x)
    return keras.Model(inp, out, name="lstm_mel")


def build_model(cfg: dict) -> keras.Model:
    """Factory: build and compile the model named in ``cfg['model']['name']``."""
    m = cfg["model"]
    T, sr = cfg["data"]["clip_length"], cfg["data"]["sample_rate"]
    name = m["name"]
    if name == "cnn1d":
        model = build_cnn1d(T, m["cnn1d"])
    elif name == "cnn2d_mel":
        model = build_cnn2d_mel(T, sr, m["mel"], m["cnn2d_mel"])
    elif name == "lstm_mel":
        model = build_lstm_mel(T, sr, m["mel"], m["lstm_mel"])
    else:
        raise ValueError(f"Unknown model: {name}")

    model.compile(
        optimizer=keras.optimizers.Adam(cfg["training"]["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.AUC(curve="PR", name="pr_auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model
