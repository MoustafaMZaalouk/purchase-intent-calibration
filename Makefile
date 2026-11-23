PY ?= python3
DATA := data/online_shoppers_intention.csv

.PHONY: all data train eval test lint

all: eval test

$(DATA):
	$(PY) scripts/download_data.py

data: $(DATA)

train: $(DATA)
	$(PY) scripts/run_experiment.py
	$(PY) scripts/run_experiment.py --include-page-values

eval: train
	$(PY) scripts/make_plots.py
	$(PY) scripts/make_plots.py --include-page-values

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src scripts tests
