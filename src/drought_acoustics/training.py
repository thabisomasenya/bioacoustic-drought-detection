"""Model training.

Training strategy
-----------------
* **Optimiser**: Adam (adaptive per-parameter learning rates, robust default for
  CNNs), lr from config, halved by ``ReduceLROnPlateau`` when validation loss stalls.
* **Loss**: binary cross-entropy on a sigmoid output (P(drought)).
* **Imbalance**: drought clips outnumber cut clips ~1.6:1, so the loss is
  re-weighted with ``class_weight='balanced'`` computed on the *training split*.
  This is preferred to undersampling (it discards no data) and to oversampling
  (augmentation already supplies variety).
* **Early stopping** on validation ROC-AUC (threshold-free, so it is insensitive
  to the class prior) with best-weight restoration. This is the main guard
  against over-fitting a ~500k-parameter network to ~2,000 clips.
* Every epoch is logged to ``history.csv`` and the best model is checkpointed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .augmentation import make_dataset
from .data import load_or_build_manifest, load_waveforms, select_noise, select_signals
from .models import build_model
from .preprocessing import AmplitudeScaler, filter_signals
from .utils import get_logger

log = get_logger(__name__)


@dataclass
class PreparedData:
    """Filtered (not yet scaled) arrays plus their metadata."""
    signals: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    noise: pd.DataFrame
    X_noise: np.ndarray


def prepare_data(cfg: dict) -> PreparedData:
    manifest = load_or_build_manifest(cfg)
    signals = select_signals(cfg, manifest)
    noise = select_noise(manifest)
    noise = noise[noise["noise_source"].isin(cfg["augmentation"]["noise_sources"])]
    noise = noise.reset_index(drop=True)
    X_all = load_waveforms(cfg, manifest)
    pos = {p: i for i, p in enumerate(manifest["path"])}
    X = filter_signals(X_all[[pos[p] for p in signals["path"]]], cfg)
    X_noise = filter_signals(X_all[[pos[p] for p in noise["path"]]], cfg)
    log.info("Prepared %d plant clips (%d drought / %d cut) and %d noise clips",
             len(X), int(signals["label"].sum()), int((signals["label"] == 0).sum()), len(X_noise))
    return PreparedData(signals, X, signals["label"].to_numpy(), noise, X_noise)


def class_weights(y: np.ndarray, mode: str | None) -> dict | None:
    if mode != "balanced":
        return None
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: float(w[0]), 1: float(w[1])}


def callbacks(cfg: dict, out_dir: Path | None, verbose: int = 0) -> list:
    t = cfg["training"]
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                         patience=t["early_stopping_patience"],
                                         restore_best_weights=True, verbose=verbose),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=t["reduce_lr_factor"],
                                             patience=t["reduce_lr_patience"], min_lr=1e-6,
                                             verbose=verbose),
    ]
    if out_dir is not None:
        cbs += [tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
                tf.keras.callbacks.ModelCheckpoint(str(out_dir / "model.keras"), monitor="val_auc",
                                                   mode="max", save_best_only=True)]
    return cbs


def fit(cfg: dict, data: PreparedData, split: dict[str, np.ndarray],
        noise_split: dict[str, np.ndarray], out_dir: Path | None = None,
        epochs: int | None = None, verbose: int = 2):
    """Fit scaler + model on one train/val split. Returns (model, scaler, history)."""
    seed = cfg["project"]["seed"]
    tr, va = split["train"], split["val"]

    scaler = AmplitudeScaler(cfg["preprocessing"]["scaling"]).fit(data.X[tr])
    X_tr, X_va = scaler.transform(data.X[tr]), scaler.transform(data.X[va])
    noise_bank = scaler.transform(data.X_noise[noise_split["train"]])

    use_aug = cfg["augmentation"].get("enabled", False)
    ds_tr = make_dataset(X_tr, data.y[tr], cfg, training=True,
                         noise_bank=noise_bank if use_aug else None, seed=seed)
    ds_va = make_dataset(X_va, data.y[va], cfg, training=False)

    model = build_model(cfg)
    cw = class_weights(data.y[tr], cfg["training"].get("class_weighting"))
    log.info("Training %s | train=%d val=%d | augmentation=%s | class_weight=%s",
             model.name, len(tr), len(va), use_aug, cw)
    history = model.fit(ds_tr, validation_data=ds_va,
                        epochs=epochs or cfg["training"]["epochs"],
                        class_weight=cw, callbacks=callbacks(cfg, out_dir), verbose=verbose)
    return model, scaler, history
