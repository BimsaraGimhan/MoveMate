# MoveMate Rent Model

## Overview
This project trains a supervised regression model to predict Australian weekly rent by suburb and dwelling features. It includes:
- A CatBoost regression model optimized for categorical data
- A baseline model using median rent by (suburb, bedrooms, dwelling_type)
- Time-aware evaluation to prevent leakage
- CLI scripts for training and inference

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Place your dataset at `data/rents.csv` with these columns:
- date (ISO string, e.g. 2024-06-30)
- state (e.g. WA, VIC, NSW)
- suburb
- postcode
- bedrooms
- dwelling_type
- weekly_rent (target)

If the file is missing, you can generate a tiny synthetic sample for a dry run:
```bash
python scripts/train_model.py --generate-sample
```

## Training
```bash
python scripts/train_model.py
```

Artifacts are saved to `artifacts/`:
- `catboost_model.cbm`
- `baseline.pkl`
- `feature_schema.json`
- `metrics.json`

## Inference
Single example via arguments:
```bash
python scripts/predict_one.py \
  --suburb "Perth" \
  --state "WA" \
  --bedrooms 2 \
  --dwelling_type "unit" \
  --postcode 6000 \
  --date "2025-12-31"
```

From a CSV:
```bash
python -m src.predict --csv path/to/input.csv
```

From a JSON string:
```bash
python -m src.predict --json '{"date":"2025-12-31","state":"WA","suburb":"Perth","postcode":"6000","bedrooms":2,"dwelling_type":"unit"}'
```

## REST API
Start the server:
```bash
make serve
```

Generate OpenAPI (Swagger) spec:
```bash
make openapi
```

Generate Swagger YAML:
```bash
make swagger
```

Endpoints:
- `GET /health`
- `POST /predict` (single record)
- `POST /predict-batch` (array of records)

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"date":"2025-12-31","state":"WA","suburb":"Perth","postcode":"6000","bedrooms":2,"dwelling_type":"unit"}'
```

## Frontend (React)
The frontend lives in `frontend/` and uses Vite.

Install dependencies:
```bash
cd frontend
npm install
```

Run the dev server:
```bash
npm run dev
```

Optional API base override:
```bash
VITE_API_BASE=http://localhost:8000 npm run dev
```

## Notes on leakage prevention
- Features are computed in date order by suburb.
- Lagged rent features use only prior periods (shifted by 1).
- The test set is the most recent 20% of dates.
- Cross-validation uses time-based splits on training dates.
- Missing numeric values are filled with the training fold median only.

## Metrics
The primary metric is MAE (mean absolute error) in dollars/week. Both CV and holdout MAE are stored in `artifacts/metrics.json`.
# MoveMate
# MoveMate
