.PHONY: setup setup-local data eda-a evaluate predict release monitor select tune tune-tabpfn lint format test check

# Renku: run once per session (pip installs don't survive a shutdown).
# --no-deps so pip never replaces the conda packages from environment.yml.
setup:
	pip install --no-deps -e .
	pre-commit install

# Outside Renku (plain venv): pull runtime + dev deps from pyproject.toml
setup-local:
	pip install -e ".[dev]"
	pre-commit install

# Stage the challenge data into data/raw (add ARGS=--link to symlink instead of copy)
data:
	python -m brpmm.prepare $(ARGS)

# Process A analysis -> outputs/process_A/eda/REPORT.md (+ data/processed features)
eda-a:
	python scripts/eda_process_a.py $(ARGS)

# Every method on the shared CV protocol -> findings report
evaluate:
	python -m brpmm.evaluate --process A B

# LEADERBOARD FILE: final models on all calibration batches -> outputs/<time>_A-ensemble_B-pca_*.csv
# (never submits; submitting is brpmm.submission.submit)
predict:
	python -m brpmm.predict

# Batch-release assistant (industry use): one action per test batch + cut-off cost table
release:
	python -m brpmm.release

# Accepted-only monitoring (the brief's second solution) -> outputs/<time>_monitor.csv/.json
monitor:
	python scripts/monitor.py

# Evidence for the submitted A model: final selection (~2 min), nested tuning, extra views
# and synthetic rejects (~30 min); TabPFN separately (pip install -e ".[tabpfn]", TABPFN_TOKEN)
select:
	python scripts/select_final_a.py
tune:
	python scripts/tune_process_a.py --part hpo
	python scripts/tune_process_a.py --part combine
	python scripts/tune_process_a.py --part clean
	python scripts/tune_process_a.py --part augment
tune-tabpfn:
	python scripts/tune_process_a.py --part tabpfn

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest

check: lint test
