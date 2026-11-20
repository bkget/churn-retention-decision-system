PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
HF_BASE := https://huggingface.co/datasets/aai510-group1/telco-customer-churn/resolve/main
IBM_DIR := data/raw/ibm_extended
# Optional: path to an existing copy of the original 21-column CSV
KAGGLE_CSV ?= ../DataTalks_Projects/telco-customer-churn-prediction/data/Telco-Customer-Churn.csv

.PHONY: help setup data features train app test lint readme clean

help:
	@echo "make setup   - create .venv and install project + dev + llm extras"
	@echo "make data    - download IBM extended dataset (+ copy 21-col fallback if available)"
	@echo "make train   - train LR baseline + tuned XGBoost, SHAP drivers, ROI threshold -> artifacts/"
	@echo "make app     - launch the Streamlit decision app"
	@echo "make test    - run unit tests"
	@echo "make lint    - ruff check"
	@echo "make readme  - regenerate README.md from artifacts/ (run after make train)"

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev,llm]"
	@test -f .env || cp .env.example .env
	@echo "Setup complete. Activate with: source $(BIN)/activate"

data:
	@mkdir -p $(IBM_DIR)
	@for f in train validation test; do \
		if [ ! -f $(IBM_DIR)/$$f.csv ]; then \
			echo "Downloading $$f.csv"; \
			curl -fsSL -o $(IBM_DIR)/$$f.csv $(HF_BASE)/$$f.csv; \
		fi; \
	done
	@if [ ! -f data/raw/Telco-Customer-Churn.csv ] && [ -f "$(KAGGLE_CSV)" ]; then \
		cp "$(KAGGLE_CSV)" data/raw/Telco-Customer-Churn.csv && echo "Copied 21-column fallback"; \
	fi
	@wc -l $(IBM_DIR)/*.csv

features: data
	$(BIN)/python -m src.data

train: data
	$(BIN)/python -m src.train
	$(BIN)/python -m src.explain
	$(BIN)/python -m src.roi

app:
	$(BIN)/streamlit run app.py

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check app.py src tests scripts

readme:
	$(BIN)/python scripts/build_readme.py

clean:
	rm -rf artifacts data/processed reports/figures .pytest_cache
