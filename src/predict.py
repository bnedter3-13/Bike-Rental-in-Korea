

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
import yaml

from preprocess import engineer_features, load_raw_data, split_features_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
   
    return X.reindex(columns=model.feature_names_in_, fill_value=0)


def predict(raw_df: pd.DataFrame, model) -> pd.Series:
    df = engineer_features(raw_df)
    if "Bike_Count" in df.columns:
        X, _ = split_features_target(df)
    else:
        X = df
    X = align_to_model(X, model)
    return pd.Series(model.predict(X), index=raw_df.index, name="Predicted_Bike_Count")


def main(config_path: str = "config.yaml", input_path: str = None, n: int = 5):
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root / config_path)

    model_path = repo_root / config["output"]["models_dir"] / config["output"]["model_filename"]
    logger.info("Loading model from %s", model_path)
    model = load_model(model_path)

    input_path = input_path or (repo_root / config["data"]["raw_path"])
    logger.info("Loading raw input from %s", input_path)
    raw_df = load_raw_data(str(input_path), encoding=config["data"]["encoding"])

    sample = raw_df.head(n)
    predictions = predict(sample, model)

    result = sample[["Date", "Hour"]].copy()
    if "Bike_Count" in sample.columns:
        result["Actual_Bike_Count"] = sample["Bike_Count"]
    result["Predicted_Bike_Count"] = predictions.round(1)
    logger.info("Predictions:\n%s", result.to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None, help="Path to a raw CSV with SeoulBikeData.csv columns")
    parser.add_argument("--n", type=int, default=5, help="Number of rows to predict")
    args = parser.parse_args()
    main(args.config, args.input, args.n)
