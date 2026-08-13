

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from preprocess import load_features, split_features_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main(config_path: str = "config.yaml"):
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root / config_path)

    data_path = repo_root / config["data"]["raw_path"]
    logger.info("Loading and engineering features from %s", data_path)
    df = load_features(str(data_path), encoding=config["data"]["encoding"])
    X, y = split_features_target(df)

    split_cfg = config["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )
    y_train_log = np.log1p(y_train)
    logger.info("Train rows: %d, test rows: %d, features: %d", len(X_train), len(X_test), X.shape[1])

    logger.info("Training Linear Regression (on log-transformed target)")
    lr = LinearRegression()
    lr.fit(X_train, y_train_log)
    lr_pred = np.expm1(lr.predict(X_test))
    lr_metrics = evaluate(y_test, lr_pred)
    logger.info("Linear Regression metrics: %s", lr_metrics)

    rf_cfg = config["model"]["random_forest"]
    logger.info("Training Random Forest with params: %s", rf_cfg)
    rf = RandomForestRegressor(**rf_cfg)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = evaluate(y_test, rf_pred)
    logger.info("Random Forest metrics: %s", rf_metrics)

    selected_name = "random_forest"
    selected_model = rf

    output_cfg = config["output"]
    models_dir = repo_root / output_cfg["models_dir"]
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / output_cfg["model_filename"]
    with open(model_path, "wb") as f:
        pickle.dump(selected_model, f)
    logger.info("Saved selected model (%s) to %s", selected_name, model_path)

    metrics = {
        "model_version": config["model"]["version"],
        "selected_model": selected_name,
        "models": {
            "linear_regression": lr_metrics,
            "random_forest": rf_metrics,
        },
    }
    metrics_path = models_dir / output_cfg["metrics_filename"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
