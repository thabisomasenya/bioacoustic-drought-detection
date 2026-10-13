"""Dataset manifest and waveform loading.

Data versioning
---------------
Raw audio is never modified. Instead, a *manifest* (one row per .wav) records
each file's SHA-256 hash, label, plant group and quality checks. The hash of all
file hashes is the **dataset fingerprint**: every run stores it, so any change to
the underlying audio (a file added, removed or altered) is detectable, and the
decoded-waveform cache is keyed by it (content-addressed caching, the same idea
that tools such as DVC use).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .utils import get_logger

log = get_logger(__name__)

FILENAME_RE = re.compile(r"id_(?P<plant>\d+)_sound_(?P<sound>\d+)\.wav$")
CLIP_THRESHOLD = 32767 / 32768  # int16 full scale


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(cfg: dict) -> pd.DataFrame:
    """Scan the raw directory, validate every file and return the manifest."""
    dcfg = cfg["data"]
    raw_dir = Path(dcfg["raw_dir"])
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    folders = {**{k: ("signal", v) for k, v in dcfg["classes"].items()},
               **{k: ("noise", v) for k, v in dcfg["noise_sources"].items()}}
    rows = []
    for folder, (kind, meta) in folders.items():
        files = sorted((raw_dir / folder).glob("*.wav"))
        if not files:
            log.warning("No .wav files found in %s", raw_dir / folder)
        for path in files:
            m = FILENAME_RE.search(path.name)
            row = {
                "path": str(path.relative_to(raw_dir)),
                "folder": folder,
                "kind": kind,
                "label": meta["label"] if kind == "signal" else -1,
                "species": meta["species"] if kind == "signal" else "none",
                "noise_source": meta if kind == "noise" else "none",
                "plant_id": m["plant"] if m else None,
                "sound_id": int(m["sound"]) if m else None,
                # Plant IDs restart per folder, so the group key is folder-qualified.
                "group": f"{folder}/{m['plant']}" if m else f"{folder}/unknown",
                "sha256": _sha256(path),
            }
            row.update(_quality_checks(path, dcfg))
            rows.append(row)

    df = pd.DataFrame(rows)
    n_bad = int((~df["valid"]).sum())
    log.info("Manifest: %d files (%d invalid) across %d folders", len(df), n_bad, len(folders))
    return df


def _quality_checks(path: Path, dcfg: dict) -> dict:
    """Integrity checks standing in for 'missing data' handling on audio."""
    out = {"sample_rate": None, "n_samples": None, "channels": None,
           "peak_abs": np.nan, "rms": np.nan, "clipped_frac": np.nan,
           "valid": False, "issue": ""}
    try:
        info = sf.info(str(path))
        x, sr = sf.read(str(path), dtype="int16")
    except Exception as exc:  # unreadable / truncated file
        out["issue"] = f"unreadable: {type(exc).__name__}"
        return out
    x = x.astype(np.float32) / 32768.0
    out.update(sample_rate=sr, n_samples=len(x), channels=info.channels,
               peak_abs=float(np.abs(x).max()), rms=float(np.sqrt(np.mean(x**2))),
               clipped_frac=float(np.mean(np.abs(x) >= CLIP_THRESHOLD)))
    issues = []
    if sr != dcfg["sample_rate"]:
        issues.append(f"sample_rate={sr}")
    if len(x) != dcfg["clip_length"]:
        issues.append(f"length={len(x)}")
    if info.channels != 1:
        issues.append(f"channels={info.channels}")
    if not np.isfinite(x).all():
        issues.append("non-finite")
    if out["rms"] == 0:
        issues.append("silent")
    out["issue"] = ";".join(issues)
    out["valid"] = not issues
    return out


def dataset_fingerprint(manifest: pd.DataFrame) -> str:
    """Hash of all file hashes (order-independent): the dataset version ID."""
    blob = "".join(sorted(manifest["sha256"])).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_or_build_manifest(cfg: dict, rebuild: bool = False) -> pd.DataFrame:
    path = Path(cfg["project"]["artifacts_dir"]) / "manifest.csv"
    if path.exists() and not rebuild:
        return pd.read_csv(path, dtype={"plant_id": str})
    df = build_manifest(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Manifest written to %s (fingerprint %s)", path, dataset_fingerprint(df))
    return df


def load_waveforms(cfg: dict, manifest: pd.DataFrame) -> np.ndarray:
    """Decode all valid clips to a float32 array (N, clip_length) in [-1, 1).

    The array is cached as .npy keyed by the dataset fingerprint, so decoding
    5,000+ files happens once per dataset version.
    """
    cache = (Path(cfg["project"]["artifacts_dir"]) / "cache"
             / f"waveforms_{dataset_fingerprint(manifest)}_{len(manifest)}.npy")
    if cache.exists():
        return np.load(cache)
    raw_dir = Path(cfg["data"]["raw_dir"])
    X = np.stack([sf.read(str(raw_dir / p), dtype="int16")[0] for p in manifest["path"]])
    X = X.astype(np.float32) / 32768.0
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, X)
    log.info("Decoded %d clips -> %s", len(X), cache)
    return X


def select_signals(cfg: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    """Valid plant clips restricted to the configured species."""
    m = manifest[(manifest["kind"] == "signal") & manifest["valid"]]
    return m[m["species"].isin(cfg["data"]["species"])].reset_index(drop=True)


def select_noise(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest[(manifest["kind"] == "noise") & manifest["valid"]].reset_index(drop=True)
