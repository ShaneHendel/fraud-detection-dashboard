"""
Feature engineering for fraud detection.
Transforms raw transaction data into model-ready features.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioral and contextual features from raw transaction data.
    
    Returns a new DataFrame with engineered features appended.
    """
    df = df.copy()
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

    # --- Temporal features ---
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # --- Amount features ---
    # Per-cardholder statistics
    card_stats = df.groupby("cc_num")["amt"].agg(["mean", "median", "std"]).reset_index()
    card_stats.columns = ["cc_num", "card_amt_mean", "card_amt_median", "card_amt_std"]
    card_stats["card_amt_std"] = card_stats["card_amt_std"].fillna(1.0)
    df = df.merge(card_stats, on="cc_num", how="left")

    df["amt_zscore"] = (df["amt"] - df["card_amt_mean"]) / df["card_amt_std"].clip(lower=1.0)
    df["amt_to_median_ratio"] = df["amt"] / df["card_amt_median"].clip(lower=1.0)
    df["log_amt"] = np.log1p(df["amt"])

    # --- Geographic features ---
    df["distance_to_merchant"] = _haversine(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    df["log_distance"] = np.log1p(df["distance_to_merchant"])

    # --- Category features ---
    cat_fraud_rate = df.groupby("category")["is_fraud"].mean().reset_index()
    cat_fraud_rate.columns = ["category", "category_fraud_rate"]
    df = df.merge(cat_fraud_rate, on="category", how="left")

    # --- Velocity features ---
    df["tx_frequency_1h"] = _rolling_tx_count(df, window_hours=1)
    df["tx_frequency_24h"] = _rolling_tx_count(df, window_hours=24)

    # --- Age at transaction ---
    if "dob" in df.columns:
        df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25

    # --- Population features ---
    df["log_city_pop"] = np.log1p(df["city_pop"])

    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature columns used for modeling."""
    return [
        "amt", "log_amt", "hour", "day_of_week", "is_weekend", "is_night",
        "amt_zscore", "amt_to_median_ratio",
        "distance_to_merchant", "log_distance",
        "category_fraud_rate",
        "tx_frequency_1h", "tx_frequency_24h",
        "age", "log_city_pop",
    ]


def _haversine(lat1, lon1, lat2, lon2) -> pd.Series:
    """Vectorized haversine distance in kilometers."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _rolling_tx_count(df: pd.DataFrame, window_hours: int = 1) -> pd.Series:
    """
    Count transactions per card within a rolling time window.
    Approximation: groups by card and counts within time buckets.
    """
    df = df.copy()
    df["_time_bucket"] = df["trans_date_trans_time"].dt.floor(f"{window_hours}h")
    counts = df.groupby(["cc_num", "_time_bucket"]).cumcount() + 1
    return counts
