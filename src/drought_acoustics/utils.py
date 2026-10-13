"""Cross-cutting helpers: seeding, logging, run folders and provenance."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "drought_acoustics") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def add_file_handler(logger: logging.Logger, path: str | Path) -> None:
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline touches.

    ``enable_op_determinism`` makes TensorFlow kernels deterministic on CPU so a
    re-run with the same config reproduces the same weights (at some speed cost).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # pragma: no cover - older TF
            os.environ["TF_DETERMINISTIC_OPS"] = "1"


def make_run_dir(runs_dir: str | Path, experiment: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir) / f"{stamp}_{experiment}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def environment_info() -> dict:
    """Capture the software environment so results can be traced and reproduced."""
    import librosa
    import scipy
    import sklearn
    import tensorflow as tf

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "librosa": librosa.__version__,
        "git_commit": git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def write_json(obj, path: str | Path) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def read_json(path: str | Path):
    with open(path) as fh:
        return json.load(fh)
