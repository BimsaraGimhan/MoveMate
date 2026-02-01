You are an expert ML engineer. Create a complete, runnable Python project to train a NEW machine learning regression model that predicts Australian weekly rent by suburb and dwelling features (Australia-focused relocation app).

GOALS
- Train a supervised regression model from tabular data.
- Use time-aware evaluation (avoid leakage).
- Save the trained model and preprocessing artifacts.
- Provide a simple CLI and a small inference script to test predictions.

TECH REQUIREMENTS
- Python 3.11+
- Use pandas, numpy, scikit-learn, catboost (preferred because it handles categorical features well)
- Also implement a baseline model: median weekly rent by (suburb, bedrooms, dwelling_type) from training data
- Metric: MAE (mean absolute error) in dollars/week
- Reproducible: fixed random seeds
- Clear logging and readable code

DATA
- Assume there will be a local CSV at: data/rents.csv
- The CSV will contain these columns (exact names):
  - date: ISO string date like "2024-06-30" (quarter end or month end)
  - state: string like "WA", "VIC", "NSW"
  - suburb: string suburb name
  - postcode: string or int
  - bedrooms: int
  - dwelling_type: string (e.g., "house", "unit", "apartment")
  - weekly_rent: float (TARGET)
- Handle missing values robustly:
  - drop rows missing weekly_rent
  - fill missing categorical with "unknown"
  - fill missing numeric with median (computed from training fold only)

FEATURE ENGINEERING (NO LEAKAGE)
- Parse date to datetime
- Create year and quarter features
- Sort by (suburb, date) and create a lag feature:
  - suburb_rent_lag1 = previous period weekly_rent for that suburb (groupby suburb shift(1))
- Drop rows where lag is missing
- Optional additional feature:
  - rolling_suburb_median_4 = rolling median of weekly_rent for previous 4 periods (shifted by 1) per suburb
- Ensure lag/rolling features are computed without peeking into future data

SPLITTING
- Use time-based splitting:
  - Sort all rows by date
  - Use last 20% of dates as test set (holdout) OR implement TimeSeriesSplit with 5 splits
- Print both cross-val MAE and final holdout MAE

MODEL
- CatBoostRegressor with loss_function="MAE"
- Use categorical features: suburb, state, dwelling_type (postcode can be treated as categorical too if you want)
- Reasonable defaults: iterations around 1500, learning_rate 0.05, depth 8
- Early stopping on eval set
- Output feature importance table (top 20)

PROJECT STRUCTURE
Create the following files:

1) pyproject.toml (or requirements.txt) with dependencies
2) README.md with:
   - overview
   - setup instructions
   - how to run training
   - how to run inference
   - notes about time-split and leakage prevention
3) src/config.py: constants (paths, seed)
4) src/data.py:
   - load_data(path) -> DataFrame
   - clean_and_validate(df) -> DataFrame (checks columns, types)
5) src/features.py:
   - add_time_features(df)
   - add_lag_features(df)
6) src/baseline.py:
   - train_baseline(df_train) -> object (dict tables)
   - predict_baseline(baseline, df) -> np.array
7) src/train.py:
   - main training entrypoint
   - trains baseline + catboost
   - evaluates MAE
   - saves artifacts to artifacts/:
     - catboost_model.cbm
     - baseline.pkl
     - feature_schema.json
     - metrics.json
8) src/predict.py:
   - loads artifacts
   - predicts on one example JSON input or a small CSV
9) scripts/train_model.py: calls src.train main
10) scripts/predict_one.py: example prediction call

SAVING ARTIFACTS
- Use joblib for baseline dict and any sklearn preprocessors if used
- Save CatBoost model using its native save_model
- Save metrics.json with MAE numbers for baseline and catboost on CV and holdout

QUALITY
- Include type hints
- Include docstrings
- Include basic error handling (missing file, bad columns)
- Keep code clean and modular
- Do not hardcode OS-specific paths

DELIVERABLE
Output ALL files with code content. Make sure the project can be run with:
- pip install -r requirements.txt
- python scripts/train_model.py
- python scripts/predict_one.py --suburb "Perth" --state "WA" --bedrooms 2 --dwelling_type "unit" --postcode 6000 --date "2025-12-31"
Implement argparse for scripts.

IMPORTANT
- Do not fetch data from the internet.
- Assume the user will provide data/rents.csv.
- If data/rents.csv is missing, show a helpful message explaining expected columns and give a tiny synthetic sample generator option (optional).
