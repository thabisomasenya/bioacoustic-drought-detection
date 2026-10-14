"""Command-line entry point.

Examples
--------
    python -m drought_acoustics.cli manifest
    python -m drought_acoustics.cli eda
    python -m drought_acoustics.cli train  -e configs/experiments/cnn1d_aug.yaml
    python -m drought_acoustics.cli train  -e configs/experiments/cnn1d_aug.yaml --cv
    python -m drought_acoustics.cli tune   -e configs/experiments/cnn1d_aug.yaml
    python -m drought_acoustics.cli train  -e configs/experiments/cnn1d_aug.yaml \\
        --params artifacts/tuning/cnn1d_aug/best_params.json
    python -m drought_acoustics.cli compare
    python -m drought_acoustics.cli report --run runs/<run>   # re-render report.html
    python -m drought_acoustics.cli all            # manifest -> eda -> 4 experiments -> compare

    # any config key can be overridden without editing files:
    python -m drought_acoustics.cli train -e configs/experiments/cnn1d_aug.yaml \\
        --set training.epochs=5 --set data.species=[tomato]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from .config import load_config  # noqa: E402

EXPERIMENTS = ["cnn1d_aug", "cnn1d_noaug", "cnn2d_mel_aug", "lstm_mel_aug"]
EXP_DIR = Path("configs/experiments")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="drought_acoustics", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["manifest", "eda", "train", "tune", "compare", "report", "all"])
    ap.add_argument("-e", "--experiment", type=Path, help="experiment YAML (overrides default)")
    ap.add_argument("-c", "--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config key, e.g. training.epochs=5")
    ap.add_argument("--params", type=Path, metavar="JSON",
                    help="apply tuned hyperparameters, e.g. artifacts/tuning/cnn1d_aug/best_params.json")
    ap.add_argument("--cv", action="store_true", help="grouped K-fold CV instead of holdout")
    ap.add_argument("--run", type=Path, action="append", default=[],
                    help="run folder(s) for the report stage")
    ap.add_argument("-v", "--verbose", type=int, default=2, help="Keras verbosity (0-2)")
    args = ap.parse_args(argv)

    from . import pipeline as pl  # heavy imports (TensorFlow) only after arg parsing

    overrides = list(args.set)
    if args.params:
        best = json.loads(args.params.read_text())["best_params"]
        overrides = [f"{k}={json.dumps(v)}" for k, v in best.items()] + overrides
    cfg = load_config(args.experiment, overrides, base=args.config)
    if args.stage == "manifest":
        pl.stage_manifest(cfg)
    elif args.stage == "eda":
        pl.stage_eda(cfg)
    elif args.stage == "train":
        (pl.run_cv if args.cv else pl.run_holdout)(cfg, verbose=args.verbose)
    elif args.stage == "tune":
        pl.stage_tune(cfg)
    elif args.stage == "report":
        for run in args.run:
            pl.rebuild_report(run)
    elif args.stage == "compare":
        pl.stage_compare(cfg, EXPERIMENTS)
        pl.stage_compare_cv(cfg, EXPERIMENTS)
    elif args.stage == "all":
        pl.stage_manifest(cfg)
        pl.stage_eda(cfg)
        tuned = Path(cfg["project"]["artifacts_dir"]) / "tuning" / "cnn1d_aug" / "best_params.json"
        for exp in EXPERIMENTS:
            ov = list(args.set)
            if exp.startswith("cnn1d") and tuned.exists():   # tuned params shared by 1D variants
                best = json.loads(tuned.read_text())["best_params"]
                ov = [f"{k}={json.dumps(v)}" for k, v in best.items()] + ov
            pl.run_holdout(load_config(EXP_DIR / f"{exp}.yaml", ov, base=args.config),
                           verbose=args.verbose)
        pl.stage_compare(cfg, EXPERIMENTS)


if __name__ == "__main__":
    main()
