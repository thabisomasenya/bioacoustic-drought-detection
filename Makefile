# One-command workflow. Each target is a pipeline stage; `make all` reproduces every result.
PY ?= python
CLI = $(PY) -m drought_acoustics.cli
EXP = configs/experiments
BEST = artifacts/tuning/cnn1d_aug/best_params.json

.PHONY: setup test manifest eda tune train ablation baselines cv compare all clean

setup:      ## create the environment (run inside an activated Python 3.10 venv)
	pip install -r requirements.txt && pip install -e .

test:       ## unit tests (filters, SNR mixing, leakage checks, model I/O)
	$(PY) -m pytest -q

manifest:   ## validate + hash every .wav (data versioning)
	$(CLI) manifest

eda:        ## exploratory figures -> artifacts/eda/
	$(CLI) eda

tune:       ## hyperparameter search on the validation split only
	$(CLI) tune -e $(EXP)/cnn1d_aug.yaml

train:      ## proposed model: 1D CNN + SNR augmentation
	$(CLI) train -e $(EXP)/cnn1d_aug.yaml --params $(BEST)

ablation:   ## same model without augmentation
	$(CLI) train -e $(EXP)/cnn1d_noaug.yaml --params $(BEST)

baselines:  ## Deliverable 1 models: 2D Mel-CNN and Mel-LSTM
	$(CLI) train -e $(EXP)/cnn2d_mel_aug.yaml
	$(CLI) train -e $(EXP)/lstm_mel_aug.yaml

cv:         ## 5-fold plant-grouped CV, all models on identical folds
	$(CLI) train -e $(EXP)/cnn1d_aug.yaml --params $(BEST) --cv --verbose 0
	$(CLI) train -e $(EXP)/cnn2d_mel_aug.yaml --cv --verbose 0
	$(CLI) train -e $(EXP)/lstm_mel_aug.yaml --cv --verbose 0
	$(CLI) train -e $(EXP)/cnn1d_noaug.yaml --params $(BEST) --cv --verbose 0

compare:    ## comparison table + figures -> artifacts/comparison/
	$(CLI) compare

all:        ## everything, resumable (same as scripts/run_all.sh)
	bash scripts/run_all.sh

clean:      ## remove generated outputs (raw data is never touched)
	rm -rf artifacts/eda artifacts/comparison artifacts/tuning runs
