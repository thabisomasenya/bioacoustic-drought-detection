# Progress log: Deliverable 4 implementation

**Session 1: Tue 13 Oct 2026, 17:30–21:45 (SAST).** Pipeline built, holdout runs, 1D CNN CV.
**Session 2: Wed 14 Oct 2026, from 09:27 (SAST).** CV completed for all models, statistical
comparison added, and the Deliverable 4 report drafted (Claude Doc).
**Status:** all experiments are complete and committed. The report draft covers all 8 sections.

---

## 1. Status at a glance

| Item | State |
|---|---|
| Data audit, manifest, data versioning | Done: 5,477 files, 0 invalid, 0 duplicates, fingerprint `8332402651c23ab8` |
| EDA figures (6) | Done: `artifacts/eda/` |
| Modular package (13 modules) + CLI + YAML configs | Done |
| Unit tests | Done: 34 passing (`make test`) |
| Representative plant-grouped split (v2) | Done (v1 kept as evidence in `artifacts/decision_log/`) |
| Hyperparameter tuning | Done: `artifacts/tuning/cnn1d_aug/` |
| Holdout runs (4 models) + HTML reports | Done: `runs/20261013-2*` |
| 5-fold grouped CV, all 4 models | Done |
| Paired comparison (corrected t-test + paired cluster bootstrap) | Done: `artifacts/comparison/` |
| README results | Final |
| Deliverable 4 report | Drafted in a Claude Doc, all 8 sections + references; awaiting your review |

## 2. What was done and why (chronological)

1. **Read the brief, the rubric, Deliverable 1 and the dataset.**
   *Why:* the rubric PDF omits "Model Development" (8 marks) and lists Visualization as 8. The
   brief (authoritative) totals 50 with Visualization at 6. Deliverable 1 committed us to:
   Butterworth 20–100 kHz, Mel-spectrograms, 2D CNN vs LSTM, a stratified 70/15/15 split,
   precision and macro-F1, the Δt lead time, and three severity classes. Every later choice is
   traced back to these commitments.

2. **Profiled the raw data before writing any model code.**
   Findings: every clip is 1,001 samples (peak ± 500), so `input_shape=(1001, 1)` is correct.
   There are 170 plants. 82–89% of plant-clip power lies in 20–100 kHz, which validates the D1
   band. Clips per plant range from 1 to 148, so the split must be grouped by plant. Noise comes
   from only 1 (Empty Pot) and 4 (Greenhouse) recording sessions.
   *Why:* design decisions should follow from the data.

3. **Built the pipeline as a package, not a notebook.**
   Modules: `config` → `data` → `preprocessing` → `splitting` → `augmentation` → `models` →
   `training` → `evaluation` → `visualization` → `reporting` → `pipeline` → `cli`.
   *Why:* this targets the rubric's Pipeline & Automation (8) and Code Quality (5) criteria.
   Each stage is a pure function of the YAML config, so any result can be regenerated from the
   `config.yaml` saved in its run folder.

4. **Data versioning without extra tooling.** Every file gets a SHA-256 hash, and a dataset
   fingerprint goes into every run. The waveform cache is keyed by that fingerprint.
   *Why:* it delivers the DVC idea (content-addressed data and detectable change) with no new
   dependency.

5. **Kept the D1 baselines and ran them through the identical pipeline.** The Mel front-end is a
   Keras layer using a Librosa filterbank, so every model receives exactly the same augmented
   waveforms.
   *Why:* the claim that the 1D CNN is a justified upgrade has to be measured, not asserted.

6. **Augmentation implemented as on-the-fly in-band SNR mixing in `tf.data`**, with
   stateless-seeded random ops.
   *Why:* this makes the project's stated Data Strategy concrete and reproducible.

7. **Evaluation protocol.** The threshold is chosen on validation only. CIs come from a
   plant-level bootstrap. Results are broken down per species and per plant, and an SNR
   robustness sweep uses held-out noise. A majority-class baseline is included.
   *Why:* the dataset is small and clustered, so naive clip-level metrics overstate certainty.

8. **First full run exposed a split problem (see challenge C1), so the splitter was redesigned
   and tuning was re-run.**

9. **Container restart killed the first tuning run (challenge C2), so tuning and CV were made
   resumable.**

10. **Holdout results showed the 2D Mel-CNN slightly ahead of the 1D CNN with overlapping CIs
    (challenge C3), so paired 5-fold CV was queued for all four models.**

**Session 2**

11. **Resumed the runner.** It skipped finished work and continued the 2D Mel-CNN CV after its
    second fold, then ran Mel-LSTM CV and 1D CNN no-augmentation CV.
12. **Added formal model comparison.** This includes the Nadeau–Bengio corrected resampled t-test
    on fold scores and a paired plant-level bootstrap of the AUC difference on pooled out-of-fold
    predictions. Per-species out-of-fold AUC was also added.
13. **Verified the reference code's origin.** The Cell paper's STAR Methods point to a Zenodo
    release (doi:10.5281/zenodo.7612742) of the "CNN binary classifier training function", whose
    description matches `context/CNN_sounds_classifier.py`.
14. **Added a `report` stage** that re-renders a run's HTML report without retraining. Added 7
    evaluation unit tests (34 in total) and polished the figures.
15. **Drafted the full Deliverable 4 report**, section by section, with figures and two
    interactive charts.

---

## 3. Results

### 3.1 Hyperparameter search (1D CNN, validation ROC-AUC, 30-epoch budget)

| kernel | dropout | batch-norm | lr | best epoch | val AUC |
|---|---|---|---|---|---|
| **25** | **0.5** | **yes** | **1e-3** | 28 | **0.903** (selected) |
| 25 | 0.3 | yes | 1e-3 | 21 | 0.902 |
| 25 | 0.3 | no | 3e-4 | 28 | 0.899 |
| 9 | 0.5 | yes | 1e-3 | 26 | 0.888 |
| 9 | 0.5 | no | 3e-4 | 24 | 0.883 |
| 9 | 0.5 | yes | 3e-4 | 30 | 0.815 |

Finding: a 25-sample kernel (50 µs, about 2.75 cycles at 55 kHz) beats the reference 9-sample
kernel (18 µs) in every pairing. Wider first-layer filters resolve the click's frequency content
better.

### 3.2 Holdout test set (550 clips, 26 plants; threshold tuned on validation)

| Model | Macro-F1 | ROC-AUC [95% CI] | Bal. acc. | Plant AUC | AUC @ 0 dB | AUC @ −5 dB |
|---|---|---|---|---|---|---|
| 1D CNN + aug (proposed) | 0.644 | 0.692 [0.53–0.85] | 0.647 | 0.631 | 0.657 | 0.640 |
| 1D CNN, no aug | 0.616 | 0.662 [0.51–0.82] | 0.622 | 0.650 | 0.613 | 0.597 |
| 2D Mel-CNN + aug (D1) | 0.671 | 0.728 [0.53–0.86] | 0.681 | 0.687 | 0.708 | 0.648 |
| Mel-LSTM + aug (D1) | 0.594 | 0.649 [0.52–0.79] | 0.610 | 0.631 | 0.634 | 0.645 |
| Majority class | 0.379 | 0.500 | 0.500 | – | – | – |

### 3.3 5-fold plant-grouped CV (identical folds for every model)

| Model | ROC-AUC mean ± sd | Macro-F1 mean ± sd | Pooled OOF AUC [95% CI] | Plant AUC | Tomato / tobacco AUC |
|---|---|---|---|---|---|
| 1D CNN + aug | **0.781 ± 0.047** | **0.695 ± 0.038** | 0.767 [0.70–0.82] | 0.752 | 0.809 / 0.593 |
| 1D CNN, no aug | 0.773 ± 0.048 | 0.692 ± 0.041 | 0.745 [0.67–0.81] | 0.701 | 0.774 / 0.592 |
| 2D Mel-CNN + aug | 0.757 ± 0.038 | 0.679 ± 0.045 | 0.713 [0.63–0.78] | 0.662 | 0.720 / 0.597 |
| Mel-LSTM + aug | 0.705 ± 0.064 | 0.640 ± 0.044 | 0.701 [0.64–0.75] | 0.673 | 0.712 / 0.558 |

Paired tests, 1D CNN + aug minus the other model:
- vs no-aug: ΔAUC +0.007 per fold (p = 0.57); pooled +0.021 [−0.029, +0.074].
- vs 2D Mel-CNN: ΔAUC +0.024 per fold, 4/5 folds (p = 0.27); pooled +0.054 [−0.029, +0.137].
- vs Mel-LSTM: ΔAUC +0.076 per fold, 5/5 folds (p = 0.059; macro-F1 p = 0.028); pooled +0.066 [+0.004, +0.124].

### 3.4 What the evidence says

- **Every model beats the majority baseline.** The signal is real but moderate. This is
  consistent with the original study, where an SVM separated drought from cut sounds at about 70%
  accuracy. Verify this figure against the Cell paper before citing it.
- **Augmentation helps, especially under noise.** At equal architecture, AUC is +0.03 clean,
  +0.04 at 0 dB and +0.04 at −5 dB.
- **The holdout is high-variance.** The tuned model scored 0.91 on validation and 0.69 on test,
  and the 95% CIs span about 0.3 AUC. Which plants land in which split matters a lot. CV is the
  primary estimate.
- **Tobacco is harder than tomato** (holdout AUC 0.57 vs 0.69 for the 1D CNN). There are few
  drought-tobacco clips (275), and species is partly confounded with class.
- **1D vs 2D, settled by CV:** the 1D CNN is ahead on every summary but not significantly.
  Report claim: it performs at least as well as the spectrogram model, with a consistent but
  unconfirmed edge. It is significantly better than the LSTM.

---

## 4. Challenges and how they were resolved

| # | Challenge | Resolution | Evidence |
|---|---|---|---|
| C1 | The first split put only 2 cut-tomato plants in test (cut = 23% of test vs 38% overall); test AUC CI 0.62–0.89. `StratifiedGroupKFold` balances plant counts but not clip counts when plants contribute 1–148 clips. | The splitter searches 40 seeded candidate splits and keeps the most representative one: L1 distance of class × species clip mix, size deviation, and a ≥3-plants-per-stratum penalty. Same for CV folds. Tuning re-run on the new validation set to avoid test leakage. Unit test added. | `artifacts/decision_log/split_v1_unbalanced/`, `docs/DECISIONS.md` D3 |
| C2 | A container restart mid-tuning lost a background job. | Tuning and CV now write each trial/fold to disk and skip completed ones; `scripts/run_all.sh` skips finished stages. | `pipeline.py` (`stage_tune`, `run_cv`) |
| C3 | The holdout ranks the 2D Mel-CNN above the 1D CNN, with overlapping CIs. | No model changes after seeing test results (that would overfit to test). Paired 5-fold CV for all models on identical folds. | `artifacts/comparison/` |
| C4 | Robustness evaluation crashed on negative SNR (negative RNG seed). | Seed offset fixed. | `evaluation.py` |
| C5 | Spectrogram figure frame count didn't match the model (librosa `center=True` vs TF `stft`). | `center=False` in the figure; axis relabelled "frame start". | `visualization.py` |
| C6 | Cannot compute D1's Δt lead time or three severity classes. | Documented as a dataset limitation: no timestamps, binary labels. Carried to future work. | `README.md` §5, `docs/DECISIONS.md` |

---

## 5. What could be done better (honest critique)

**Methodology**
1. **Tune with cross-validation, not one validation split.** The 0.91 → 0.69 validation-test gap
   shows a single grouped validation set (~20 plants) is a noisy selection signal. Nested
   grouped CV, or selecting on mean CV AUC, would give more trustworthy hyperparameters.
2. **Tune the baselines too.** Only the 1D CNN was tuned. The 2D Mel-CNN and LSTM use defaults,
   which biases the comparison toward the 1D CNN. Even so, the 2D CNN was competitive.
3. **Raise the epoch budget.** One tuning trial peaked at its final epoch (30/30), and CV fold 0
   used all 60 epochs. Some runs may be under-trained. Try 100 epochs with patience 15.
4. **Weight plants equally.** A few plants contribute over 100 clips and dominate both the loss
   and the test metrics. Sample weights ∝ 1/clips-per-plant would make the model learn
   "drought" rather than "plant 35".
5. ~~**Use a formal paired test** for model comparison.~~ *Done in session 2: corrected resampled t-test plus paired cluster bootstrap.*
6. **Threshold stability.** The validation-tuned threshold transfers imperfectly across plant
   sets. Plant-level aggregation with a calibrated threshold (e.g. Platt scaling on CV
   out-of-fold scores) is the deployable decision rule.
7. **Model capacity.** The 1D CNN ends in Flatten → Dense (about 1.04M parameters with k=25, versus 28k for the 2D Mel-CNN). A
   global-average-pooling head would cut parameters and possibly over-fitting. This should be
   tested via CV *before* looking at any test set, as a pre-registered variant.
8. **Species.** Train species-aware models, or add species as an auxiliary output, and report
   the confound explicitly.

**Engineering**
9. **Compute.** CPU-only with 2 cores made the full study take about 4 hours. On the MacBook,
   `tensorflow-metal` or a Colab GPU would cut this by 5–10×. TF op determinism also costs speed;
   it can be disabled for exploration and enabled for final runs.
10. **Experiment tracking.** JSON/CSV run folders work, but MLflow (or Weights & Biases) would
    give a searchable UI across runs at no extra code cost.
11. **Data versioning.** The manifest+fingerprint approach could be swapped for DVC with a remote
    so the 22 MB dataset is pulled by hash on any machine.
12. **CI.** Add a GitHub Actions workflow that runs `make test` on every push.
13. **Process note.** The first two runs were committed before git was initialised, so their
    `run_info.json` shows `git_commit: null`. All runs from session 2 onward record the commit.

---

## 6. Next steps

1. **Review the report draft** in the Claude Doc. Each section maps to one rubric criterion.
2. **Commit `context/CNN_sounds_classifier.py`** from your Mac terminal:
   `git add context/CNN_sounds_classifier.py && git commit -m "Add reference model"`.
   Then push to a private GitHub repo and add the link to Section 7.3.
3. **Optional v2 experiment**, pre-registered before looking at any test set: improvements 1–4
   in section 5.

To re-run everything from scratch: `make test && bash scripts/run_all.sh` (resumable).

---

## 7. File map (where things are)

| Path | What |
|---|---|
| `configs/default.yaml`, `configs/experiments/` | All parameters |
| `src/drought_acoustics/` | Pipeline code |
| `tests/` | Unit tests |
| `scripts/run_all.sh` | Full, resumable reproduction |
| `docs/DECISIONS.md` | Design decisions with evidence (feeds the report) |
| `docs/PROGRESS_LOG.md` | This file |
| `artifacts/data_report.json`, `artifacts/manifest.csv` | Data audit and versioning |
| `artifacts/eda/` | 6 EDA figures |
| `artifacts/tuning/cnn1d_aug/` | Tuning results and best parameters |
| `artifacts/comparison/` | Cross-model tables and figures |
| `artifacts/decision_log/` | Evidence of the v1 split problem |
| `runs/<timestamp>_<experiment>/` | Per-run config, metrics, predictions, figures, `report.html`, model |
| `artifacts/cache/` | Regenerable caches. `plantsounds_snapshot.tgz` was only used to copy the data to the cloud workspace and can be deleted. |
