PYTHON ?= python3
VENV_DIR ?= .venv

.PHONY: help venv install train train-sample predict serve openapi swagger

help:
	@echo "Targets:"
	@echo "  venv          Create virtual environment"
	@echo "  install       Install dependencies"
	@echo "  train         Train models (expects data/rents.csv)"
	@echo "  train-sample  Train models with synthetic data if missing"
	@echo "  predict       Run example prediction"
	@echo "  serve         Run the REST API server"
	@echo "  openapi       Generate OpenAPI (Swagger) spec"
	@echo "  swagger       Generate Swagger YAML spec"

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install:
	$(VENV_DIR)/bin/python -m pip install -r requirements.txt

train:
	$(VENV_DIR)/bin/python -m src.train

train-sample:
	$(VENV_DIR)/bin/python -m src.train --generate-sample

predict:
	$(VENV_DIR)/bin/python -m src.predict --suburb "Perth" --state "WA" --bedrooms 3 --dwelling_type "unit" --postcode 6164 --date "2025-12-31"

serve:
	$(VENV_DIR)/bin/python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

openapi:
	$(VENV_DIR)/bin/python scripts/generate_openapi.py

swagger:
	$(VENV_DIR)/bin/python scripts/generate_openapi_yaml.py
