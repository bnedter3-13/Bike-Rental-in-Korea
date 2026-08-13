

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "Date",
    "Bike_Count",
    "Hour",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Visibility",
    "Dew_Point",
    "Solar_Radiation",
    "Rainfall",
    "Snowfall",
    "Seasons",
    "Holiday",
    "Functioning_Day",
]


def load_raw_data(csv_path: str, encoding: str = "cp1252") -> pd.DataFrame:
   
    df = pd.read_csv(csv_path, encoding=encoding)
    df.columns = RAW_COLUMNS
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.day_name()

    df = pd.get_dummies(
        df,
        columns=["Seasons", "Holiday", "Functioning_Day", "Weekday"],
        drop_first=True,
    )

    df["Day_Type"] = np.where(
        (df.get("Holiday_No Holiday", 0) == 0) | (df["Date"].dt.dayofweek >= 5),
        "Leisure",
        "Work",
    )
    df = pd.get_dummies(df, columns=["Day_Type"], drop_first=True)

    df.drop(columns=["Date"], inplace=True)
    if "Dew_Point" in df.columns:
        df.drop(columns=["Dew_Point"], inplace=True)

    df["Temp_sq"] = df["Temperature"] ** 2
    df["Humidity_sq"] = df["Humidity"] ** 2
    df["Temp_x_Humidity"] = df["Temperature"] * df["Humidity"]
    df["Temp_x_Solar"] = df["Temperature"] * df["Solar_Radiation"]

    hour_dummies = pd.get_dummies(df["Hour"], prefix="Hour", drop_first=True)
    df = pd.concat([df.drop(columns=["Hour"]), hour_dummies], axis=1)

    return df


def load_features(csv_path: str, encoding: str = "cp1252") -> pd.DataFrame:
    return engineer_features(load_raw_data(csv_path, encoding))


def split_features_target(df: pd.DataFrame, target: str = "Bike_Count"):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
