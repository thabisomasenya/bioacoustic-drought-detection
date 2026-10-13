"""drought_acoustics: bioacoustic drought-stress detection pipeline.

Classifies 2 ms, 500 kHz ultrasonic plant emissions (Khait et al., 2023) as
mechanical stress (cut, class 0) or drought stress (dry, class 1).

Modules
-------
config         YAML loading, deep-merge of experiment overrides, dotted overrides
utils          seeding, logging, run directories, provenance capture
data           dataset manifest (SHA-256 data versioning) and waveform loading
preprocessing  DC removal, Butterworth band-pass, train-fitted amplitude scaling
splitting      plant-grouped, class x species stratified train/val/test splits
augmentation   tf.data input pipeline with in-band SNR noise mixing
models         1D CNN (raw waveform), 2D Mel-CNN and Mel-LSTM baselines
training       model fitting with callbacks and class weighting
tuning         random search over a declared hyperparameter space
evaluation     metrics, threshold selection, SNR robustness, plant-level scores
visualization  EDA and results figures
reporting      self-contained HTML run report
cli            command-line entry point that chains the stages
"""

__version__ = "1.0.0"
