# Public-sector recourse audit
#
#   make setup    # create .venv, install dependencies, download the data
#   make test     # automated correctness checks
#   make quick    # ~15 s smoke test of the full pipeline
#   make audit    # full audit (a few minutes)
#   make app      # launch the Streamlit explorer
#
# Run `make help` to list every target.

PYTHON ?= python3
VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
STAMP  := $(VENV)/.installed

.DEFAULT_GOAL := help
.PHONY: help setup install data test verify quick audit app clean clean-all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: install data  ## Install dependencies and download the data (run this first)

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

# Re-install whenever requirements.txt changes.
$(STAMP): requirements.txt | $(VENV)/bin/python
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(STAMP)

install: $(STAMP)  ## Create .venv and install requirements.txt

data: $(STAMP)  ## Download UCI Adult from OpenML into data/raw/ (cached)
	$(PY) -c "from src.data_loader import load_adult; df = load_adult(); print('data/raw/adult.csv:', df.shape)"

test: $(STAMP)  ## Run the automated correctness tests
	$(PY) -m pytest tests -q

verify: $(STAMP)  ## Print a readable walkthrough: data, model, recourse examples
	$(PY) scripts/verify.py

quick: $(STAMP)  ## Smoke test: tiny end-to-end run (~15 s)
	$(PY) run_audit.py --quick

audit: $(STAMP)  ## Full audit, writes data/processed/ and figures/
	$(PY) run_audit.py

app: $(STAMP)  ## Launch the Streamlit app
	$(VENV)/bin/streamlit run app/app.py

clean:  ## Remove caches and trained models (keeps results and data)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f data/processed/*.joblib

clean-all: clean  ## Also remove .venv and the downloaded data
	rm -rf $(VENV) data/raw
