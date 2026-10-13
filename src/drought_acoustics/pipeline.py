"""Pipeline stages. Each stage is a pure function of the config.

    manifest -> eda -> train (holdout or grouped CV) -> compare
                 \\-> tune (validation only, never touches test)

Every training run gets its own folder under ``runs/`` containing everything
needed to audit or reproduce it: the resolved config, environment and dataset
fingerprint, split membership, the trained model and scaler, per-epoch history,
per-clip predictions, metrics, figures and an HTML report.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluation as ev
from . import visualization as viz
from .config import config_hash, deep_merge, save_config, set_dotted
from .data import dataset_fingerprint, load_or_build_manifest, load_waveforms, select_signals
from .preprocessing import filter_signals
from .reporting import write_report
from .splitting import grouped_cv, grouped_holdout, split_noise, summarise_split
from .training import PreparedData, fit, prepare_data
from .utils import (add_file_handler, environment_info, get_logger, make_run_dir, read_json,
                    set_global_seed, write_json)

log = get_logger("drought_acoustics.pipeline")


# ------------------------------------------------------------------ manifest
def stage_manifest(cfg: dict, rebuild: bool = True) -> pd.DataFrame:
    manifest = load_or_build_manifest(cfg, rebuild=rebuild)
    sig = manifest[manifest["kind"] == "signal"]
    report = {
        "dataset_fingerprint": dataset_fingerprint(manifest),
        "n_files": len(manifest),
        "n_invalid": int((~manifest["valid"]).sum()),
        "invalid_examples": manifest.loc[~manifest["valid"], ["path", "issue"]].head(20)
                                    .to_dict("records"),
        "duplicate_audio_files": int(manifest["sha256"].duplicated().sum()),
        "per_folder": manifest.groupby("folder").agg(
            files=("path", "size"), plants=("group", "nunique"),
            median_rms=("rms", "median"), pct_with_clipping=("clipped_frac",
                                                             lambda s: float((s > 0).mean() * 100)))
            .reset_index().to_dict("records"),
        "class_balance": sig.groupby("label").size().to_dict(),
    }
    write_json(report, Path(cfg["project"]["artifacts_dir"]) / "data_report.json")
    log.info("Data report: %d files, %d invalid, %d duplicates, fingerprint %s",
             report["n_files"], report["n_invalid"], report["duplicate_audio_files"],
             report["dataset_fingerprint"])
    return manifest


# ----------------------------------------------------------------------- EDA
def stage_eda(cfg: dict) -> list[Path]:
    out = Path(cfg["project"]["artifacts_dir"]) / "eda"
    manifest = load_or_build_manifest(cfg)
    X_all = load_waveforms(cfg, manifest)
    X_filt = filter_signals(X_all, cfg)
    sr = cfg["data"]["sample_rate"]
    bp = cfg["preprocessing"]["bandpass"]
    signals = select_signals(cfg, manifest)
    rng = np.random.default_rng(cfg["project"]["seed"])
    figs = [viz.plot_class_distribution(manifest, out / "01_class_distribution.png"),
            viz.plot_clips_per_plant(signals, out / "02_clips_per_plant.png")]

    folders = ["Tomato Cut", "Tomato Dry", "Tobacco Cut", "Tobacco Dry", "Empty Pot",
               "Greenhouse Noises"]
    pick = [int(rng.choice(np.flatnonzero(manifest["folder"] == f))) for f in folders]
    figs.append(viz.plot_example_waveforms(X_all[pick], X_filt[pick], folders, sr,
                                           out / "03_example_waveforms.png"))
    groups, colors = {}, {}
    for f, c in zip(folders, [viz.BLUE, viz.ORANGE, "#86b6ef", "#f4a37f", viz.AQUA, viz.VIOLET]):
        groups[f] = X_all[(manifest["folder"] == f).to_numpy()]
        colors[f] = c
    figs.append(viz.plot_mean_psd(groups, colors, sr, (bp["low_hz"], bp["high_hz"]),
                                  out / "04_mean_psd.png"))
    dry_idx = int(rng.choice(np.flatnonzero(manifest["folder"] == "Tomato Dry")))
    figs.append(viz.plot_spectrogram_vs_waveform(
        X_filt[dry_idx], sr, cfg["model"]["mel"],
        "One drought click, three representations", out / "05_representations.png"))
    noise_idx = np.flatnonzero((manifest["kind"] == "noise").to_numpy())
    figs.append(viz.plot_snr_examples(X_filt[dry_idx], X_filt[noise_idx], sr,
                                      out / "06_snr_augmentation.png"))
    log.info("EDA figures written to %s", out)
    return figs


# --------------------------------------------------------------------- train
def _run_info(cfg: dict, data: PreparedData) -> dict:
    manifest = load_or_build_manifest(cfg)
    return {"experiment": cfg.get("experiment", "default"), "model": cfg["model"]["name"],
            "config_hash": config_hash(cfg), "dataset_fingerprint": dataset_fingerprint(manifest),
            "n_signals": len(data.signals), "n_noise": len(data.noise),
            "environment": environment_info()}


def _pred_frame(data: PreparedData, idx: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    df = data.signals.iloc[idx][["path", "group", "species", "label"]].reset_index(drop=True)
    df["prob"] = prob
    return df


def _evaluate_split(cfg, data, model, scaler, split, noise_split):
    """Threshold on val, then all test-time analyses."""
    X_va = scaler.transform(data.X[split["val"]])
    X_te = scaler.transform(data.X[split["test"]])
    p_va, p_te = ev.predict(model, X_va, cfg), ev.predict(model, X_te, cfg)
    thr = ev.choose_threshold(data.y[split["val"]], p_va, cfg["evaluation"]["threshold"])
    test_pred = _pred_frame(data, split["test"], p_te)
    test_pred["pred"] = (p_te >= thr).astype(int)
    plants, plant_m = ev.plant_level(test_pred, thr)
    noise_te = data.noise.iloc[noise_split["test"]]
    rob = ev.robustness(model, X_te, data.y[split["test"]],
                        scaler.transform(data.X_noise[noise_split["test"]]),
                        noise_te["noise_source"].to_numpy(), cfg, thr)
    metrics = {
        "threshold": thr,
        "val": ev.compute_metrics(data.y[split["val"]], p_va, thr),
        "test": ev.compute_metrics(data.y[split["test"]], p_te, thr),
        "test_at_0.5": ev.compute_metrics(data.y[split["test"]], p_te, 0.5),
        "test_ci": ev.bootstrap_ci(test_pred, thr, cfg["evaluation"]["bootstrap_iterations"],
                                   cfg["project"]["seed"]),
        "test_per_species": ev.per_species(test_pred, thr),
        "plant_level": plant_m,
        "majority_baseline": ev.majority_baseline(data.y[split["train"]], data.y[split["test"]]),
    }
    return metrics, test_pred, plants, rob


def run_holdout(cfg: dict, verbose: int = 2) -> Path:
    set_global_seed(cfg["project"]["seed"])
    run_dir = make_run_dir(cfg["project"]["runs_dir"], cfg.get("experiment", "default"))
    add_file_handler(log, run_dir / "run.log")
    save_config(cfg, run_dir / "config.yaml")
    data = prepare_data(cfg)
    info = _run_info(cfg, data)
    write_json(info, run_dir / "run_info.json")

    split = grouped_holdout(data.signals, cfg)
    noise_split = split_noise(data.noise, cfg)
    split_tbl = summarise_split(data.signals, split)
    split_tbl.to_csv(run_dir / "split_summary.csv", index=False)
    write_json({k: sorted(data.signals["group"].iloc[v].unique()) for k, v in split.items()},
               run_dir / "split_groups.json")
    log.info("Split (clips): %s", {k: len(v) for k, v in split.items()})

    model, scaler, _ = fit(cfg, data, split, noise_split, out_dir=run_dir, verbose=verbose)
    write_json(scaler.to_dict(), run_dir / "scaler.json")
    metrics, test_pred, plants, rob = _evaluate_split(cfg, data, model, scaler, split, noise_split)
    write_json(metrics, run_dir / "metrics.json")
    test_pred.to_csv(run_dir / "test_predictions.csv", index=False)
    plants.to_csv(run_dir / "test_plant_scores.csv", index=False)
    rob.to_csv(run_dir / "robustness.csv", index=False)

    fig_dir = run_dir / "figures"
    hist = pd.read_csv(run_dir / "history.csv")
    thr = metrics["threshold"]
    figs = [viz.plot_training_curves(hist, fig_dir / "training_curves.png", info["experiment"]),
            viz.plot_confusion(metrics["test"], fig_dir / "confusion_matrix.png"),
            viz.plot_roc_pr({info["experiment"]: (test_pred["label"], test_pred["prob"])},
                            fig_dir / "roc_pr.png", viz.MODEL_COLORS, viz.MODEL_LABELS),
            viz.plot_prob_hist(test_pred, thr, fig_dir / "score_distribution.png"),
            viz.plot_plant_scores(plants, thr, fig_dir / "plant_scores.png"),
            viz.plot_robustness({info["experiment"]: rob}, "roc_auc", fig_dir / "robustness_auc.png",
                                viz.MODEL_COLORS, viz.MODEL_LABELS)]
    headline = pd.DataFrame({k: metrics[k] for k in ["test", "test_at_0.5", "val",
                                                       "plant_level", "majority_baseline"]}).T
    cols = ["n", "threshold", "macro_f1", "roc_auc", "pr_auc", "balanced_accuracy",
            "precision_drought", "recall_drought", "specificity", "mcc"]
    tables = {"Headline metrics": headline[cols],
              "Per species (test)": pd.DataFrame(metrics["test_per_species"]).T[cols],
              "Robustness (all noise)": rob[rob["noise_source"] == "all"]
              .set_index("snr_db")[["macro_f1", "roc_auc", "recall_drought", "specificity"]],
              "Split composition": split_tbl.set_index("split")}
    write_report(run_dir, metrics, figs, tables, info)
    t = metrics["test"]
    log.info("TEST | macro-F1 %.3f | ROC-AUC %.3f | bal-acc %.3f | thr %.2f | run %s",
             t["macro_f1"], t["roc_auc"], t["balanced_accuracy"], thr, run_dir)
    return run_dir


def run_cv(cfg: dict, verbose: int = 0) -> Path:
    """Grouped K-fold CV: a more stable estimate than one holdout on 170 plants."""
    set_global_seed(cfg["project"]["seed"])
    name = cfg.get("experiment", "default") + "_cv"
    # Resume an unfinished CV run with the same config instead of starting over.
    unfinished = [p for p in sorted(Path(cfg["project"]["runs_dir"]).glob(f"*_{name}"))
                  if not (p / "cv_summary.json").exists() and (p / "run_info.json").exists()
                  and read_json(p / "run_info.json")["config_hash"] == config_hash(cfg)]
    run_dir = unfinished[-1] if unfinished else make_run_dir(cfg["project"]["runs_dir"], name)
    add_file_handler(log, run_dir / "run.log")
    save_config(cfg, run_dir / "config.yaml")
    data = prepare_data(cfg)
    if not unfinished:
        write_json(_run_info(cfg, data), run_dir / "run_info.json")
    noise_split = split_noise(data.noise, cfg)
    rows, oof = [], []
    for k, split in enumerate(grouped_cv(data.signals, cfg)):
        fold_m, fold_p = run_dir / f"fold{k}_metrics.json", run_dir / f"fold{k}_predictions.csv"
        if fold_m.exists() and fold_p.exists():
            log.info("Fold %d already complete, loading", k)
            rows.append(read_json(fold_m))
            oof.append(pd.read_csv(fold_p))
            continue
        set_global_seed(cfg["project"]["seed"] + k)
        model, scaler, hist = fit(cfg, data, split, noise_split, verbose=verbose)
        p_va = ev.predict(model, scaler.transform(data.X[split["val"]]), cfg)
        p_te = ev.predict(model, scaler.transform(data.X[split["test"]]), cfg)
        thr = ev.choose_threshold(data.y[split["val"]], p_va, cfg["evaluation"]["threshold"])
        m = ev.compute_metrics(data.y[split["test"]], p_te, thr)
        m.update(fold=k, epochs_run=len(hist.history["loss"]),
                 test_plants=data.signals["group"].iloc[split["test"]].nunique())
        rows.append(m)
        fp = _pred_frame(data, split["test"], p_te)
        fp["fold"], fp["threshold"] = k, thr
        oof.append(fp)
        write_json(m, fold_m)
        fp.to_csv(fold_p, index=False)
        log.info("Fold %d | macro-F1 %.3f | ROC-AUC %.3f", k, m["macro_f1"], m["roc_auc"])
    folds = pd.DataFrame(rows)
    folds.to_csv(run_dir / "cv_folds.csv", index=False)
    oof_df = pd.concat(oof, ignore_index=True)
    oof_df.to_csv(run_dir / "oof_predictions.csv", index=False)
    keys = ["macro_f1", "roc_auc", "pr_auc", "balanced_accuracy", "recall_drought",
            "specificity", "mcc"]
    summary = {k: {"mean": float(folds[k].mean()), "std": float(folds[k].std())} for k in keys}
    write_json(summary, run_dir / "cv_summary.json")
    viz.plot_roc_pr({cfg.get("experiment", "cv"): (oof_df["label"], oof_df["prob"])},
                    run_dir / "figures" / "oof_roc_pr.png", viz.MODEL_COLORS, viz.MODEL_LABELS)
    log.info("CV | macro-F1 %.3f +/- %.3f | ROC-AUC %.3f +/- %.3f",
             summary["macro_f1"]["mean"], summary["macro_f1"]["std"],
             summary["roc_auc"]["mean"], summary["roc_auc"]["std"])
    return run_dir


# ---------------------------------------------------------------------- tune
def stage_tune(cfg: dict) -> Path:
    """Random search on the validation split. The test split is never touched."""
    tcfg = cfg["tuning"]
    out = Path(cfg["project"]["artifacts_dir"]) / "tuning" / cfg.get("experiment", "default")
    out.mkdir(parents=True, exist_ok=True)
    space = tcfg["search_space"]
    grid = [dict(zip(space, combo)) for combo in itertools.product(*space.values())]
    random.Random(cfg["project"]["seed"]).shuffle(grid)
    grid = grid[: tcfg["max_trials"]]

    set_global_seed(cfg["project"]["seed"])
    data = prepare_data(cfg)
    split = grouped_holdout(data.signals, cfg)
    noise_split = split_noise(data.noise, cfg)

    # Resumable: completed trials are appended to disk and skipped on restart.
    partial = out / "tuning_results.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    done = {tuple(r[k] for k in space) for r in rows}
    for i, params in enumerate(grid):
        if tuple(params.values()) in done:
            log.info("Trial %d/%d already done, skipping", i + 1, len(grid))
            continue
        trial = deep_merge(cfg, {})
        for k, v in params.items():
            set_dotted(trial, k, v)
        set_global_seed(cfg["project"]["seed"])
        _, _, hist = fit(trial, data, split, noise_split, epochs=tcfg["epochs"], verbose=0)
        h = hist.history
        best = int(np.argmax(h["val_auc"]))
        rows.append({"trial": i, **params, "best_epoch": best + 1,
                     "val_auc": h["val_auc"][best], "val_loss": h["val_loss"][best]})
        log.info("Trial %d/%d %s -> val AUC %.4f", i + 1, len(grid), params, h["val_auc"][best])
        pd.DataFrame(rows).to_csv(partial, index=False)
    res = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    res.to_csv(out / "tuning_results.csv", index=False)
    best = {k: (v.item() if hasattr(v, "item") else v)
            for k, v in res.iloc[0][list(space)].to_dict().items()}
    write_json({"best_params": best, "val_auc": float(res.iloc[0]["val_auc"])},
               out / "best_params.json")
    log.info("Best params: %s (val AUC %.4f)", best, res.iloc[0]["val_auc"])
    return out


# ------------------------------------------------------------------- compare
def latest_runs(runs_dir: str | Path, experiments: list[str]) -> dict[str, Path]:
    found = {}
    for exp in experiments:
        cands = sorted(p for p in Path(runs_dir).glob(f"*_{exp}")
                       if (p / "metrics.json").exists())
        if cands:
            found[exp] = cands[-1]
    return found


def stage_compare(cfg: dict, experiments: list[str]) -> Path:
    out = Path(cfg["project"]["artifacts_dir"]) / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    runs = latest_runs(cfg["project"]["runs_dir"], experiments)
    rows, curves, robs = {}, {}, {}
    for exp, rd in runs.items():
        m = read_json(rd / "metrics.json")
        row = {k: m["test"][k] for k in ["macro_f1", "roc_auc", "pr_auc", "balanced_accuracy",
                                          "recall_drought", "specificity", "mcc"]}
        for k, (lo, hi) in m["test_ci"].items():
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = lo, hi
        row["plant_roc_auc"] = m["plant_level"]["roc_auc"]
        info = read_json(rd / "run_info.json")
        row["run"] = rd.name
        row["model"] = info["model"]
        rob = pd.read_csv(rd / "robustness.csv")
        robs[exp] = rob
        r = rob[rob["noise_source"] == "all"].set_index("snr_db")
        for snr in ["0", "-5"]:
            if snr in r.index.astype(str):
                row[f"roc_auc_snr{snr}"] = float(r.loc[r.index.astype(str) == snr, "roc_auc"].iloc[0])
        rows[exp] = row
        pred = pd.read_csv(rd / "test_predictions.csv")
        curves[exp] = (pred["label"], pred["prob"])
    table = pd.DataFrame(rows).T
    table.to_csv(out / "model_comparison.csv")
    viz.plot_model_comparison(table.astype({c: float for c in table.columns
                                            if c not in ("run", "model")}),
                              ["macro_f1", "roc_auc", "balanced_accuracy"],
                              out / "model_comparison.png")
    viz.plot_roc_pr(curves, out / "roc_pr_comparison.png", viz.MODEL_COLORS, viz.MODEL_LABELS)
    viz.plot_robustness(robs, "roc_auc", out / "robustness_comparison.png",
                        viz.MODEL_COLORS, viz.MODEL_LABELS)
    viz.plot_robustness(robs, "macro_f1", out / "robustness_comparison_f1.png",
                        viz.MODEL_COLORS, viz.MODEL_LABELS)
    log.info("Comparison of %d runs written to %s", len(runs), out)
    return out


def stage_compare_cv(cfg: dict, experiments: list[str]) -> Path | None:
    """Compare models across the *same* plant-grouped CV folds (paired comparison).

    Every CV run uses the identical fold assignment (same data, same seed), so
    per-fold differences between models are paired and far less noisy than
    comparing two single holdout scores.
    """
    out = Path(cfg["project"]["artifacts_dir"]) / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for exp in experiments:
        cands = sorted(p for p in Path(cfg["project"]["runs_dir"]).glob(f"*_{exp}_cv")
                       if (p / "cv_summary.json").exists())
        if cands:
            runs[exp] = cands[-1]
    if not runs:
        log.info("No completed CV runs to compare")
        return None

    keys = ["macro_f1", "roc_auc", "pr_auc", "balanced_accuracy", "recall_drought",
            "specificity", "mcc"]
    folds = {exp: pd.read_csv(rd / "cv_folds.csv").set_index("fold") for exp, rd in runs.items()}
    rows, curves = {}, {}
    for exp, f in folds.items():
        row = {}
        for k in keys:
            row[f"{k}_mean"], row[f"{k}_std"] = f[k].mean(), f[k].std()
        oof = pd.read_csv(runs[exp] / "oof_predictions.csv")
        curves[exp] = (oof["label"], oof["prob"])
        pooled = ev.compute_metrics(oof["label"].to_numpy(), oof["prob"].to_numpy())
        ci = ev.bootstrap_ci(oof, 0.5, cfg["evaluation"]["bootstrap_iterations"],
                             cfg["project"]["seed"], metrics=("roc_auc",))
        row["oof_roc_auc"], (row["oof_roc_auc_ci_low"], row["oof_roc_auc_ci_high"]) = \
            pooled["roc_auc"], ci["roc_auc"]
        plants, pm = ev.plant_level(oof, 0.5)
        row["oof_plant_roc_auc"] = pm["roc_auc"]
        row["run"] = runs[exp].name
        rows[exp] = row
    table = pd.DataFrame(rows).T
    table.to_csv(out / "cv_comparison.csv")

    ref = "cnn1d_aug"
    if ref in folds:
        paired = []
        for exp, f in folds.items():
            if exp == ref:
                continue
            for k in ["roc_auc", "macro_f1"]:
                d = folds[ref][k] - f[k]
                paired.append({"comparison": f"{ref} - {exp}", "metric": k,
                               "mean_diff": d.mean(), "std_diff": d.std(),
                               "folds_ref_better": int((d > 0).sum()), "n_folds": len(d)})
        pd.DataFrame(paired).to_csv(out / "cv_paired_differences.csv", index=False)

    viz.plot_cv_folds(folds, "roc_auc", out / "cv_folds_roc_auc.png",
                      viz.MODEL_COLORS, viz.MODEL_LABELS)
    viz.plot_roc_pr(curves, out / "cv_oof_roc_pr.png", viz.MODEL_COLORS, viz.MODEL_LABELS)
    log.info("CV comparison of %d models written to %s", len(runs), out)
    return out
