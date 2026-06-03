"""
Baseline pipeline for KMA contest topic 1:
weather and spatial-data based fire-risk analysis near power facilities.

Expected files:
    - test_hanjeon.csv: KEPCO power-facility data
    - weather.csv or weather_data.csv: optional weather data with coordinates

The script performs:
    1. Basic EDA for power-facility data
    2. Coordinate-based nearest weather matching
    3. Simple weather fire-risk feature creation
    4. Random Forest baseline training/evaluation when a target column exists
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LAT_CANDIDATES = ["lat", "latitude", "위도", "y", "asset_lat", "weather_lat"]
LON_CANDIDATES = ["lon", "lng", "longitude", "경도", "x", "asset_lon", "weather_lon"]

TARGET_CANDIDATES = [
    "target",
    "label",
    "fire",
    "fire_yn",
    "fire_risk",
    "wildfire",
    "wildfire_yn",
    "산불",
    "산불발생",
    "화재",
    "화재발생",
    "화재위험",
]

TEMP_CANDIDATES = ["temperature", "temp", "ta", "기온", "평균기온", "최고기온"]
HUMIDITY_CANDIDATES = ["humidity", "relative_humidity", "rh", "습도", "상대습도"]
RAIN_CANDIDATES = ["rain", "rainfall", "precipitation", "rn", "강수", "강수량"]
WIND_CANDIDATES = ["wind", "wind_speed", "ws", "풍속", "최대풍속", "순간풍속"]
LIGHTNING_CANDIDATES = ["lightning", "thunder", "낙뢰", "뇌전"]


def read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Failed to read {path} with common Korean/UTF-8 encodings") from last_error


def find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    for col in columns:
        col_lower = str(col).strip().lower()
        if any(candidate.lower() in col_lower for candidate in candidates):
            return col
    return None


def basic_eda(df: pd.DataFrame, name: str) -> None:
    print(f"\n[{name}] shape: {df.shape}")
    print(f"\n[{name}] columns:")
    print(pd.Series(df.columns, name="column").to_string(index=False))

    print(f"\n[{name}] head:")
    print(df.head().to_string())

    missing = df.isna().mean().sort_values(ascending=False)
    print(f"\n[{name}] missing ratio top 20:")
    print(missing.head(20).to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n[{name}] numeric summary:")
        print(df[numeric_cols].describe().T.to_string())


def add_nearest_weather(
    assets: pd.DataFrame,
    weather: pd.DataFrame,
    asset_lat_col: str,
    asset_lon_col: str,
    weather_lat_col: str,
    weather_lon_col: str,
) -> pd.DataFrame:
    assets = assets.copy()
    weather = weather.copy()

    asset_coords = assets[[asset_lat_col, asset_lon_col]].astype(float)
    weather_coords = weather[[weather_lat_col, weather_lon_col]].astype(float)

    valid_assets = asset_coords.notna().all(axis=1)
    valid_weather = weather_coords.notna().all(axis=1)

    if valid_weather.sum() == 0:
        raise ValueError("Weather data has no valid coordinate rows.")

    weather_valid = weather.loc[valid_weather].reset_index(drop=True)
    weather_radians = np.radians(
        weather_valid[[weather_lat_col, weather_lon_col]].to_numpy(dtype=float)
    )
    tree = BallTree(weather_radians, metric="haversine")

    matched = assets.copy()
    matched["nearest_weather_distance_km"] = np.nan
    matched["nearest_weather_row"] = pd.NA

    asset_radians = np.radians(asset_coords.loc[valid_assets].to_numpy(dtype=float))
    distances, indices = tree.query(asset_radians, k=1)

    earth_radius_km = 6371.0088
    matched.loc[valid_assets, "nearest_weather_distance_km"] = distances[:, 0] * earth_radius_km
    matched.loc[valid_assets, "nearest_weather_row"] = indices[:, 0]

    weather_features = weather_valid.add_prefix("weather_")
    joined_weather = weather_features.iloc[indices[:, 0]].reset_index(drop=True)
    matched_valid = matched.loc[valid_assets].reset_index(drop=True)
    matched_valid = pd.concat([matched_valid, joined_weather], axis=1)

    result = matched.copy()
    for col in matched_valid.columns:
        if col not in result.columns:
            result[col] = np.nan
    result.loc[valid_assets, matched_valid.columns] = matched_valid.to_numpy()
    return result


def add_fire_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple fire-risk features from available weather columns."""
    df = df.copy()

    temp_col = find_column(df.columns, TEMP_CANDIDATES)
    humidity_col = find_column(df.columns, HUMIDITY_CANDIDATES)
    rain_col = find_column(df.columns, RAIN_CANDIDATES)
    wind_col = find_column(df.columns, WIND_CANDIDATES)
    lightning_col = find_column(df.columns, LIGHTNING_CANDIDATES)

    if temp_col:
        df["fw_high_temp"] = pd.to_numeric(df[temp_col], errors="coerce") >= 30
    if humidity_col:
        df["fw_low_humidity"] = pd.to_numeric(df[humidity_col], errors="coerce") <= 35
    if rain_col:
        rain = pd.to_numeric(df[rain_col], errors="coerce")
        df["fw_no_rain"] = rain.fillna(0) <= 0
        df["fw_low_rain"] = rain.fillna(0) < 1
    if wind_col:
        df["fw_strong_wind"] = pd.to_numeric(df[wind_col], errors="coerce") >= 7
    if lightning_col:
        df["fw_lightning"] = pd.to_numeric(df[lightning_col], errors="coerce").fillna(0) > 0

    risk_cols = [
        col
        for col in [
            "fw_high_temp",
            "fw_low_humidity",
            "fw_no_rain",
            "fw_low_rain",
            "fw_strong_wind",
            "fw_lightning",
        ]
        if col in df.columns
    ]
    if risk_cols:
        df["fire_weather_risk_score"] = df[risk_cols].astype(float).sum(axis=1)

    return df


def build_feature_frame(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    excluded = {target_col}
    leakage_keywords = [
        "fire_date",
        "wildfire_date",
        "화재일",
        "산불일",
        "발생일",
        "사고일",
    ]
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword.lower() in col_lower for keyword in leakage_keywords):
            excluded.add(col)

    X = df.drop(columns=list(excluded), errors="ignore")
    y = df[target_col]

    if y.dtype == "object":
        y = y.astype(str).str.strip()
        positive_values = {
            "1",
            "y",
            "yes",
            "true",
            "fire",
            "wildfire",
            "발생",
            "산불",
            "화재",
            "위험",
        }
        y = y.map(lambda value: 1 if value.lower() in positive_values else 0)

    return X, y.astype(int)


def train_random_forest(df: pd.DataFrame, target_col: str) -> None:
    X, y = build_feature_frame(df, target_col)
    if y.nunique() < 2:
        print(f"\nTarget '{target_col}' has fewer than two classes. Skipping model training.")
        return

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\n[RandomForest fire-risk baseline]")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"F1-score: {f1_score(y_test, pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, proba):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hanjeon", default="test_hanjeon.csv", help="Path to KEPCO facility CSV")
    parser.add_argument("--weather", default=None, help="Optional path to weather CSV")
    parser.add_argument("--target", default=None, help="Optional fire-risk target column name")
    parser.add_argument("--output", default="baseline_joined.csv", help="Joined output CSV path")
    args = parser.parse_args()

    hanjeon_path = Path(args.hanjeon)
    if not hanjeon_path.exists():
        raise FileNotFoundError(
            f"{hanjeon_path} not found. Place test_hanjeon.csv in this folder or pass --hanjeon."
        )

    assets = read_csv_safely(hanjeon_path)
    basic_eda(assets, "power_facilities")

    model_df = assets.copy()

    weather_path = Path(args.weather) if args.weather else None
    if weather_path and weather_path.exists():
        weather = read_csv_safely(weather_path)
        basic_eda(weather, "weather")

        asset_lat = find_column(assets.columns, LAT_CANDIDATES)
        asset_lon = find_column(assets.columns, LON_CANDIDATES)
        weather_lat = find_column(weather.columns, LAT_CANDIDATES)
        weather_lon = find_column(weather.columns, LON_CANDIDATES)

        if not all([asset_lat, asset_lon, weather_lat, weather_lon]):
            raise ValueError(
                "Could not detect coordinate columns. "
                f"asset_lat={asset_lat}, asset_lon={asset_lon}, "
                f"weather_lat={weather_lat}, weather_lon={weather_lon}"
            )

        model_df = add_nearest_weather(
            assets,
            weather,
            asset_lat_col=asset_lat,
            asset_lon_col=asset_lon,
            weather_lat_col=weather_lat,
            weather_lon_col=weather_lon,
        )
        model_df = add_fire_weather_features(model_df)
        model_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\nSaved joined data: {args.output}")
    elif args.weather:
        print(f"\nWeather file not found: {args.weather}. Skipping weather join.")
        model_df = add_fire_weather_features(model_df)
    else:
        print("\nNo weather file passed. Skipping weather join.")
        model_df = add_fire_weather_features(model_df)

    target_col = args.target or find_column(model_df.columns, TARGET_CANDIDATES)
    if target_col:
        train_random_forest(model_df, target_col)
    else:
        print("\nNo fire-risk target column detected. Skipping model training.")
        print("Pass --target TARGET_COLUMN when fire labels are available.")


if __name__ == "__main__":
    main()
