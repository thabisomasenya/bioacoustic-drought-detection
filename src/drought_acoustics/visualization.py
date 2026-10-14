"""Figures for EDA, training diagnostics and results.

All figures share one visual system: fixed colours per entity (cut = blue,
drought = orange, each model keeps its colour in every chart), recessive grey
axes, direct labels where they help, and 150-dpi PNG output for the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import librosa  # noqa: E402
import librosa.display  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import welch  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from .augmentation import mix_at_snr  # noqa: E402

# Validated categorical palette (fixed order, never cycled).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"

CLASS_COLORS = {0: BLUE, 1: ORANGE}
CLASS_NAMES = {0: "Cut (mechanical)", 1: "Dry (drought)"}
NOISE_COLORS = {"empty_pot": AQUA, "greenhouse": VIOLET}
MODEL_COLORS = {"cnn1d_aug": BLUE, "cnn2d_mel_aug": ORANGE, "lstm_mel_aug": AQUA,
                "cnn1d_noaug": VIOLET}
MODEL_LABELS = {"cnn1d_aug": "1D CNN + aug (proposed)", "cnn1d_noaug": "1D CNN, no aug",
                "cnn2d_mel_aug": "2D Mel-CNN + aug (D1)", "lstm_mel_aug": "Mel-LSTM + aug (D1)"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.8,
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- EDA
def plot_class_distribution(manifest: pd.DataFrame, out: Path) -> Path:
    m = manifest[manifest["valid"]]
    order = ["Tomato Cut", "Tobacco Cut", "Tomato Dry", "Tobacco Dry", "Empty Pot",
             "Greenhouse Noises"]
    counts = m.groupby("folder").agg(clips=("path", "size"), plants=("group", "nunique"))
    counts = counts.reindex([o for o in order if o in counts.index])
    colors = [CLASS_COLORS[0] if "Cut" in f else CLASS_COLORS[1] if "Dry" in f
              else MUTED for f in counts.index]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, col, title in zip(axes, ["clips", "plants"],
                              ["Clips per folder", "Plants / recording sessions per folder"]):
        bars = ax.bar(counts.index, counts[col], color=colors, width=0.7)
        ax.bar_label(bars, padding=2, color=INK_2, fontsize=8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="x", visible=False)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    fig.text(0.01, -0.06, "Blue = class 0 (cut), orange = class 1 (drought), "
             "grey = noise (augmentation only)", color=INK_2, fontsize=8)
    return _save(fig, out)


def plot_clips_per_plant(signals: pd.DataFrame, out: Path) -> Path:
    per_plant = signals.groupby(["group", "label"]).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.logspace(0, np.log10(per_plant["n"].max() + 1), 20)
    ax.hist([per_plant.loc[per_plant["label"] == lbl, "n"] for lbl in (0, 1)], bins=bins,
            color=[CLASS_COLORS[0], CLASS_COLORS[1]], label=[CLASS_NAMES[0], CLASS_NAMES[1]],
            edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Clips recorded from one plant (log scale)")
    ax.set_ylabel("Number of plants")
    ax.set_title("Clips per plant: why the split must be grouped by plant")
    ax.legend()
    return _save(fig, out)


def plot_example_waveforms(X_raw: np.ndarray, X_filt: np.ndarray, labels: list[str],
                           sr: int, out: Path) -> Path:
    t_ms = np.arange(X_raw.shape[1]) / sr * 1e3
    fig, axes = plt.subplots(len(labels), 1, figsize=(8, 1.4 * len(labels)), sharex=True,
                             constrained_layout=True)
    for ax, xr, xf, name in zip(np.atleast_1d(axes), X_raw, X_filt, labels):
        ax.plot(t_ms, xr, color=MUTED, lw=0.8, label="raw")
        ax.plot(t_ms, xf, color=BLUE, lw=1.0, label="band-passed 20-100 kHz")
        ax.set_ylabel(name, rotation=0, ha="right", va="center", color=INK)
        ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(loc="upper right", ncol=2, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Time (ms) - 2 ms window, 1,001 samples at 500 kHz")
    fig.suptitle("Example clips: raw vs zero-phase band-passed", fontweight="bold", color=INK)
    return _save(fig, out)


def plot_mean_psd(groups: dict[str, np.ndarray], colors: dict[str, str], sr: int,
                  band: tuple[float, float], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, X in groups.items():
        f, P = welch(X, fs=sr, nperseg=256, axis=1)
        Pm = P.mean(0)
        ax.plot(f / 1e3, 10 * np.log10(Pm / Pm.max()), color=colors[name], label=name)
    ax.axvspan(band[0] / 1e3, band[1] / 1e3, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text((band[0] + band[1]) / 2e3, 5, f"pass-band {band[0]/1e3:.0f}-{band[1]/1e3:.0f} kHz",
            ha="center", va="top", color=INK_2, fontsize=8)
    ax.axvline(20, color=MUTED, ls=":", lw=1)
    ax.text(21, -45, "20 kHz: upper limit of\nhuman hearing / lossy codecs", color=INK_2, fontsize=7)
    ax.set_xlim(0, sr / 2e3)
    ax.set_ylim(-50, 6)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Mean power (dB re. peak)")
    ax.set_title("Mean power spectral density (Welch) by source")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out)


def plot_spectrogram_vs_waveform(x: np.ndarray, sr: int, mel_cfg: dict, title: str,
                                 out: Path) -> Path:
    """Shows the time-frequency trade-off that motivates the 1D CNN."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    t_ms = np.arange(len(x)) / sr * 1e3
    axes[0].plot(t_ms, x, color=BLUE, lw=0.9)
    axes[0].set_title("Raw waveform (1D CNN input)\n1,001 time steps")
    axes[0].set_xlabel("Time (ms)")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=mel_cfg["n_fft"],
                                                    hop_length=mel_cfg["hop_length"],
                                                    center=False)), ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=mel_cfg["hop_length"], x_axis="ms",
                             y_axis="linear", ax=axes[1], cmap="Blues")
    axes[1].set_title(f"Linear STFT (n_fft={mel_cfg['n_fft']})\n{S.shape[1]} frames")
    M = librosa.power_to_db(librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=mel_cfg["n_fft"], hop_length=mel_cfg["hop_length"], center=False,
        n_mels=mel_cfg["n_mels"], fmin=mel_cfg["fmin"], fmax=mel_cfg["fmax"]), ref=np.max)
    librosa.display.specshow(M, sr=sr, hop_length=mel_cfg["hop_length"], x_axis="ms",
                             y_axis="mel", fmin=mel_cfg["fmin"], fmax=mel_cfg["fmax"],
                             ax=axes[2], cmap="Blues")
    axes[2].set_title(f"Log-Mel ({mel_cfg['n_mels']} bands)\n2D-CNN / LSTM input")
    for ax in axes[1:]:
        ax.grid(False)
        ax.set_xlabel("Frame start (ms)")
    fig.suptitle(title, fontweight="bold", color=INK, y=1.12)
    return _save(fig, out)


def plot_snr_examples(x: np.ndarray, noise_bank: np.ndarray, sr: int, out: Path,
                      snrs=(None, 20, 10, 0)) -> Path:
    t_ms = np.arange(len(x)) / sr * 1e3
    fig, axes = plt.subplots(1, len(snrs), figsize=(11, 2.3), sharey=True)
    for ax, snr in zip(axes, snrs):
        y = x if snr is None else mix_at_snr(x[None], noise_bank, snr, seed=3)[0]
        ax.plot(t_ms, y, color=ORANGE if snr is not None else BLUE, lw=0.8)
        ax.set_title("clean" if snr is None else f"SNR {snr} dB")
        ax.set_xlabel("Time (ms)")
    axes[0].set_yticks([])
    fig.suptitle("Augmentation: one drought click mixed with recorded greenhouse / empty-pot noise",
                 fontweight="bold", color=INK, y=1.08)
    return _save(fig, out)


# ----------------------------------------------------------------------- results
def plot_training_curves(history: pd.DataFrame, out: Path, title: str = "") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, key, name in zip(axes, ["loss", "auc"], ["Binary cross-entropy", "ROC-AUC"]):
        ax.plot(history["epoch"] + 1, history[key], color=BLUE, label="train")
        ax.plot(history["epoch"] + 1, history[f"val_{key}"], color=ORANGE, label="validation")
        ax.set_xlabel("Epoch")
        ax.set_title(name)
    best = int(history["val_auc"].idxmax())
    axes[1].axvline(history["epoch"][best] + 1, color=MUTED, ls=":", lw=1)
    axes[1].text(history["epoch"][best] + 1.5, axes[1].get_ylim()[0] + 0.01,
                 "best (restored)", color=INK_2, fontsize=8, va="bottom")
    axes[0].legend()
    fig.suptitle(title or "Training curves", fontweight="bold", color=INK, y=1.03)
    return _save(fig, out)


def plot_confusion(m: dict, out: Path, title: str = "Test confusion matrix") -> Path:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.55 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["Cut", "Dry"])
    ax.set_yticks([0, 1], ["Cut", "Dry"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    ax.set_title(f"{title}\nthreshold = {m['threshold']:.2f}")
    return _save(fig, out)


def plot_roc_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], out: Path,
                colors: dict[str, str] | None = None, labels: dict[str, str] | None = None) -> Path:
    """``curves``: name -> (y_true, prob). Overlays several models if given."""
    colors = colors or {}
    labels = labels or {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for i, (name, (y, p)) in enumerate(curves.items()):
        c = colors.get(name, [BLUE, ORANGE, AQUA, VIOLET][i % 4])
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        from sklearn.metrics import auc, average_precision_score
        axes[0].plot(fpr, tpr, color=c, label=f"{labels.get(name, name)} (AUC {auc(fpr, tpr):.3f})")
        axes[1].plot(rec, prec, color=c,
                     label=f"{labels.get(name, name)} (AP {average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1)
    y0 = next(iter(curves.values()))[0]
    axes[1].axhline(np.mean(y0), color=MUTED, ls=":", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].set(xlabel="Recall (drought)", ylabel="Precision (drought)", title="Precision-recall")
    for ax in axes:
        ax.legend(fontsize=7, loc="lower right" if ax is axes[0] else "lower left")
    return _save(fig, out)


def plot_robustness(rob: dict[str, pd.DataFrame], metric: str, out: Path,
                    colors: dict[str, str] | None = None, labels: dict[str, str] | None = None,
                    source: str = "all") -> Path:
    colors = colors or {}
    labels = labels or {}
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, (name, df) in enumerate(rob.items()):
        d = df[df["noise_source"] == source]
        x = ["clean"] + [str(s) for s in d["snr_db"] if s != "clean"]
        y = [d.loc[d["snr_db"] == "clean", metric].iloc[0]] + list(d.loc[d["snr_db"] != "clean", metric])
        c = colors.get(name, [BLUE, ORANGE, AQUA, VIOLET][i % 4])
        ax.plot(x, y, marker="o", ms=5, color=c, label=labels.get(name, name))
    ax.set_xlabel("Test-time SNR (dB, held-out noise)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Robustness to noise ({source}): {metric.replace('_', ' ')}")
    ax.legend(fontsize=8)
    return _save(fig, out)


def plot_plant_scores(plants: pd.DataFrame, threshold: float, out: Path) -> Path:
    """Stakeholder view: one dot per test plant, ranked by predicted drought risk."""
    d = plants.sort_values("prob").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for lbl in (0, 1):
        s = d[d["label"] == lbl]
        ax.scatter(s.index, s["prob"], s=np.clip(s["n_clips"] * 4, 12, 120),
                   color=CLASS_COLORS[lbl], label=f"True: {CLASS_NAMES[lbl]}",
                   edgecolor="white", linewidth=1, zorder=3)
    ax.axhline(threshold, color=MUTED, ls="--", lw=1)
    ax.text(0, threshold + 0.02, f"alert threshold {threshold:.2f}", color=INK_2, fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Test plants, ranked by mean predicted drought probability")
    ax.set_ylabel("P(drought), mean over clips")
    ax.set_title("Plant-level drought risk (dot size = clips recorded)")
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, out)


def plot_prob_hist(pred: pd.DataFrame, threshold: float, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.linspace(0, 1, 26)
    for lbl in (0, 1):
        ax.hist(pred.loc[pred["label"] == lbl, "prob"], bins=bins, alpha=0.75,
                color=CLASS_COLORS[lbl], label=CLASS_NAMES[lbl], edgecolor="white")
    ax.axvline(threshold, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("Predicted P(drought)")
    ax.set_ylabel("Test clips")
    ax.set_title("Score distribution by true class")
    ax.legend(fontsize=8)
    return _save(fig, out)


def plot_model_comparison(table: pd.DataFrame, metrics: list[str], out: Path) -> Path:
    """Grouped bars with 95% CI whiskers. ``table`` rows = experiments."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.4), sharey=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        names = list(table.index)
        vals = table[metric].to_numpy()
        lo = table.get(f"{metric}_ci_low", pd.Series(vals, index=names)).to_numpy()
        hi = table.get(f"{metric}_ci_high", pd.Series(vals, index=names)).to_numpy()
        colors = [MODEL_COLORS.get(n, MUTED) for n in names]
        ax.bar(range(len(names)), vals, color=colors, width=0.7,
               yerr=[vals - lo, hi - vals], ecolor=INK_2, capsize=3)
        for i, (v, h) in enumerate(zip(vals, hi)):     # label above the whisker, not on it
            ax.text(i, h + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(names)), [MODEL_LABELS.get(n, n) for n in names],
                      rotation=30, ha="right", fontsize=7)
        ax.set_title(metric.replace("_", " "))
        ax.set_ylim(0, 1.08)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Test-set comparison (plant-level bootstrap 95% CI)", fontweight="bold",
                 color=INK, y=1.03)
    return _save(fig, out)


def plot_cv_folds(folds: dict[str, pd.DataFrame], metric: str, out: Path,
                  colors: dict[str, str] | None = None, labels: dict[str, str] | None = None) -> Path:
    """Paired fold-level scores: one line per CV fold across models, means as bars."""
    colors = colors or {}
    labels = labels or {}
    names = list(folds)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    means = [folds[n][metric].mean() for n in names]
    ax.bar(range(len(names)), means, color=[colors.get(n, MUTED) for n in names], width=0.6,
           alpha=0.35, zorder=1)
    fold_ids = folds[names[0]].index
    for fid in fold_ids:
        ys = [folds[n].loc[fid, metric] if fid in folds[n].index else np.nan for n in names]
        ax.plot(range(len(names)), ys, color=MUTED, lw=1, zorder=2)
    for i, n in enumerate(names):
        ax.scatter([i] * len(folds[n]), folds[n][metric], color=colors.get(n, MUTED), s=36,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(i, means[i] + 0.01, f"{means[i]:.3f}", ha="center", va="bottom", color=INK,
                fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(names)), [labels.get(n, n) for n in names], fontsize=8)
    ax.set_ylabel(metric.replace("_", " "))
    lo = min(f[metric].min() for f in folds.values())
    ax.set_ylim(max(0, lo - 0.08), 1.0)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"5-fold plant-grouped CV: {metric.replace('_', ' ')} "
                 "(bars = mean, grey lines join the same fold)")
    return _save(fig, out)
