# Bike Soul — Seoul Bike Rental Demand Prediction

Predicts hourly public bike rental demand in Seoul from weather and calendar
data (temperature, humidity, hour of day, season, holidays, etc.), comparing
a Linear Regression baseline against a Random Forest model.

## Dataset

`SeoulBikeData.csv` — 8,760 hourly records (Dec 2017–Nov 2018) of Seoul's
public bike rental system, combined with weather observations. Columns:

- `Date`, `Hour` — timestamp of the observation
- `Rented Bike Count` — target variable
- `Temperature(°C)`, `Humidity(%)`, `Wind speed (m/s)`, `Visibility (10m)`,
  `Dew point temperature(°C)`, `Solar Radiation (MJ/m2)`, `Rainfall(mm)`,
  `Snowfall (cm)` — weather features
- `Seasons`, `Holiday`, `Functioning Day` — calendar/operational features

The file is CP1252-encoded (Windows-1252), which is why it must be read with
`encoding='cp1252'` rather than plain UTF-8.

## Folder structure

```
Bike_soul.ipynb        Exploratory notebook: EDA, feature engineering, model comparison
SeoulBikeData.csv      Raw dataset
config.yaml            Paths and hyperparameters (no hardcoding in scripts)
requirements.txt       Python dependencies
src/
  preprocess.py         Data loading + feature engineering, shared by train/predict
  train.py              Trains Linear Regression and Random Forest, logs metrics, saves the best model
  predict.py            Loads the saved model and predicts on new/raw rows
models/
  model_v1.pkl           Trained model artifact (versioned)
  metrics.json            RMSE / MAE / R2 for both models
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Train (reads `config.yaml`, writes `models/model_v1.pkl` and `models/metrics.json`):

```bash
python src/train.py
```

Predict on the first few rows of a raw CSV (defaults to `SeoulBikeData.csv`):

```bash
python src/predict.py --input SeoulBikeData.csv --n 5
```

## Models used and results

Both models are trained on an 75/25 train/test split (`random_state=42`).
Linear Regression is fit on a `log1p`-transformed target with squared and
interaction terms (temperature², humidity², temperature×humidity,
temperature×solar radiation) plus one-hot-encoded hours, since demand is
non-linear and right-skewed. Random Forest is fit directly on the raw count.

| Model              | RMSE   | MAE    | R2   |
|---------------------|--------|--------|------|
| Linear Regression   | 318.2  | 203.2  | 0.75 |
| Random Forest       | 192.3  | 112.2  | 0.91 |

Random Forest is the better performer on every metric and is the model saved
as `models/model_v1.pkl`. Its top predictors are temperature, whether the
system is on a functioning day, the 6 PM hour, and humidity.

## Future improvements

- Cross-validation and hyperparameter search (e.g. `GridSearchCV`) instead of
  a single train/test split and fixed Random Forest params.
- Try gradient-boosted trees (XGBoost/LightGBM), which often outperform
  Random Forest on tabular data like this.
- Time-based train/test split (train on earlier months, test on later ones)
  to better simulate real forecasting conditions instead of a random split.
- Track experiments (e.g. MLflow) instead of a single `metrics.json`, and add
  a `predict.py` REST/CLI interface for serving.
- Add unit tests for `src/preprocess.py`'s feature engineering.
