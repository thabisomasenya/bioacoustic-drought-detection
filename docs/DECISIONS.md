# Design decision log

A record of the main implementation decisions: what was chosen, what the alternatives were,
and what evidence settled each one. Entries marked *(evidence)* point to files this
repository generates.

---

### D1. Raw-waveform 1D CNN instead of Mel-spectrogram 2D CNN / LSTM

**Context.** Deliverable 1 proposed 2D Mel-spectrograms fed to a 2D CNN and an LSTM.
**Problem.** A clip is 1,001 samples (2 ms). Any STFT must trade time resolution for
frequency resolution: `n_fft=256` gives 1.95 kHz bins, but each frame spans 0.51 ms, a quarter
of the clip, while the click itself lasts about 0.1-0.2 ms. The Mel scale models human pitch
perception (roughly linear below 1 kHz, logarithmic above). It has no physiological meaning at
20-100 kHz, and it compresses exactly the band where these emissions sit.
**Decision.** Learn the filterbank from the waveform with a 1D CNN. Keep the D1 models as
baselines, given the identical input pipeline, so that the comparison is measured rather than assumed.
*(evidence: `artifacts/eda/05_representations.png`, `artifacts/comparison/`)*

### D2. Keep the Deliverable 1 band-pass (20-100 kHz, Butterworth, order 4)

**Evidence.** On plant clips, 82-89% of power lies in 20-100 kHz, and the spectral peaks sit at
41-64 kHz (`artifacts/eda/04_mean_psd.png`).
**Implementation choices.** `sosfiltfilt` (zero phase) keeps the transient peak at sample 500.
Second-order sections avoid numerical instability.

### D3. Plant-grouped, representative splitting

**Problem.** Plants contribute 1-148 clips each, and clips from one plant are correlated.
**v1** used a single `StratifiedGroupKFold` draw. It produced a test set with only 2 cut-tomato
plants (cut clips 23% vs 38% overall) and a test ROC-AUC 95% CI of 0.62-0.89.
*(evidence: `artifacts/decision_log/split_v1_unbalanced/`)*
**v2** searches 40 seeded candidate splits and keeps the one whose class x species clip mix
best matches the full data, requiring at least 3 plants per stratum. Hyperparameter tuning was
re-run on the new validation set so that no test-set information leaks into model selection.
A unit test guards representativeness.

### D4. Imbalance: class weights, not resampling

Drought clips outnumber cut clips 1.63 : 1. `class_weight='balanced'` (computed on the
training split) keeps every clip. Oversampling would duplicate clips that augmentation already
varies, and undersampling would discard about 40% of drought data.

### D5. Train-only amplitude scaling (global RMS)

This is the audio analogue of `StandardScaler`: one factor fitted on training clips. A global
factor, rather than per-clip normalisation, preserves relative loudness, which differs between
groups: median RMS is 0.031 for tomato-cut and 0.020 for tomato-dry (`artifacts/data_report.json`).

### D6. Noise augmentation design

* Noise sources: Empty Pot (microphone/self-noise events) and Greenhouse Noises, split
  70/15/15 at file level so that test-time noise is never seen in training.
* SNR is measured **in-band**, after the band-pass filter. Because filtering is linear, mixing
  pre-filtered arrays is mathematically identical to filtering the mixture, which allows mixing
  inside `tf.data` on the fly.
* SNR ~ U(0, 20) dB, applied to 80% of training clips. Noise is randomly rolled and its polarity
  flipped, together with ±20-sample shifts and polarity flips of the signal.
* Stateless, seeded random ops make every epoch's augmentation reproducible.

### D7. Threshold chosen on validation, never on test

The sigmoid output is a score, and class weighting shifts its calibration. The operating
threshold maximises validation macro-F1 and is then frozen. Metrics at 0.5 are also reported.

### D8. Uncertainty: plant-level (cluster) bootstrap

The resampling unit is the plant, not the clip, because clips from one plant are not
independent. 1,000 resamples give 95% CIs.

### D9. Reproducibility and data versioning

SHA-256 per file and a dataset fingerprint go into every run, along with the resolved config
snapshot, config hash, git commit, package versions and seeds with TF op determinism.
Long stages (tuning, CV) are resumable after interruption.

### D10. Claims follow the paired cross-validation, not the holdout

The single holdout ranked the 2D Mel-CNN first (AUC 0.729 vs 0.692), with CIs about 0.3 wide.
Paired 5-fold CV on identical folds ranked the 1D CNN first on every metric:
- AUC 0.781 vs 0.757, ahead in 4 of 5 folds;
- pooled out-of-fold AUC +0.054, with a 95% CI of [−0.029, +0.137];
- corrected p = 0.27, so the lead is not significant.

**Decision.** The report claims the 1D CNN is at least as good as the spectrogram design, with a
consistent but statistically unconfirmed edge. It is significantly better than the LSTM. No model
was altered after its test results were seen.
*(evidence: `artifacts/comparison/cv_comparison.csv`, `cv_paired_differences.csv`,
`oof_paired_bootstrap_auc.csv`)*

### D11. Statistical tests for model comparison

Fold scores share about 75% of their training data, so a plain paired t-test is
over-confident. Two tests are used:
- **Nadeau–Bengio corrected resampled t-test** on fold differences. The variance is inflated by
  1/(k−1), with df = k−1.
- **Paired plant-level bootstrap** of the AUC difference on pooled out-of-fold predictions. The
  same clips are scored by both models, and plants are resampled.

Both are unit-tested.

### Out of scope / not implementable with this dataset

* **Δt lead time (D1 objective 5):** the dataset has no timestamps relative to visible wilting.
* **Three severity levels (D1 scope):** the labels are binary (cut vs dry).
* These are carried forward as future work requiring a longitudinal recording protocol.
