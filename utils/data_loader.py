"""
Data loading and validation utilities.
Handles both real Kaggle CSVs and synthetic data generation for demo purposes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_kaggle_data(filename: str = "fraudTest.csv") -> pd.DataFrame:
    """Load and validate a Kaggle fraud detection CSV."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. Download from Kaggle and place in data/ directory."
        )

    df = pd.read_csv(filepath, index_col=0)
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"], format="%d-%m-%Y %H:%M", dayfirst=True
    )
    df["dob"] = pd.to_datetime(df["dob"], format="%d-%m-%Y", dayfirst=True)

    required_cols = ["amt", "is_fraud", "category", "lat", "long", "merch_lat", "merch_long"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df):,} transactions | Fraud rate: {df['is_fraud'].mean():.2%}")
    return df


def generate_synthetic_data(n_transactions: int = 50_000, fraud_rate: float = 0.017) -> pd.DataFrame:
    """
    Generate synthetic transaction data matching Kaggle schema.
    Use this for demo/development when full Kaggle dataset is unavailable.
    """
    rng = np.random.default_rng(42)

    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud

    categories = [
        "grocery_pos", "gas_transport", "home", "shopping_pos", "kids_pets",
        "shopping_net", "entertainment", "food_dining", "personal_care",
        "health_fitness", "misc_net", "misc_pos", "travel",
    ]

    # Legit transactions: smaller amounts, business hours bias
    legit_amts = rng.lognormal(mean=3.0, sigma=1.0, size=n_legit).clip(1, 2000)
    legit_hours = rng.normal(loc=14, scale=4, size=n_legit).clip(0, 23).astype(int)

    # Fraud transactions: larger amounts, off-hours bias
    fraud_amts = rng.lognormal(mean=5.0, sigma=1.2, size=n_fraud).clip(10, 15000)
    fraud_hours = rng.choice(24, size=n_fraud, p=_off_hours_distribution())

    # Combine
    amts = np.concatenate([legit_amts, fraud_amts])
    hours = np.concatenate([legit_hours, fraud_hours])
    is_fraud = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)

    # Generate dates across 6 months
    base_date = pd.Timestamp("2020-01-01")
    days_offset = rng.integers(0, 180, size=n_transactions)
    timestamps = [base_date + pd.Timedelta(days=int(d), hours=int(h)) for d, h in zip(days_offset, hours)]

    # Geo: US bounding box with fraud having larger merchant distance
    lat = rng.uniform(25, 48, size=n_transactions)
    lon = rng.uniform(-125, -70, size=n_transactions)
    merch_lat = lat + rng.normal(0, 0.5, size=n_transactions)
    merch_lon = lon + rng.normal(0, 0.5, size=n_transactions)
    # Push fraud merchant locations further away
    merch_lat[-n_fraud:] += rng.uniform(1, 5, size=n_fraud) * rng.choice([-1, 1], size=n_fraud)
    merch_lon[-n_fraud:] += rng.uniform(1, 5, size=n_fraud) * rng.choice([-1, 1], size=n_fraud)

    df = pd.DataFrame({
        "trans_date_trans_time": timestamps,
        "cc_num": rng.integers(1e15, 9e15, size=n_transactions),
        "merchant": [f"merchant_{i}" for i in rng.integers(0, 500, size=n_transactions)],
        "category": rng.choice(categories, size=n_transactions),
        "amt": np.round(amts, 2),
        "first": [f"first_{i}" for i in range(n_transactions)],
        "last": [f"last_{i}" for i in range(n_transactions)],
        "gender": rng.choice(["M", "F"], size=n_transactions),
        "city": [f"city_{i}" for i in rng.integers(0, 200, size=n_transactions)],
        "state": rng.choice(["TX", "CA", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"], size=n_transactions),
        "lat": lat,
        "long": lon,
        "city_pop": rng.integers(500, 1_000_000, size=n_transactions),
        "job": [f"job_{i}" for i in rng.integers(0, 100, size=n_transactions)],
        "dob": [base_date - pd.Timedelta(days=int(d)) for d in rng.integers(7000, 25000, size=n_transactions)],
        "merch_lat": merch_lat,
        "merch_long": merch_lon,
        "is_fraud": is_fraud,
    })

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Generated {len(df):,} synthetic transactions | Fraud rate: {df['is_fraud'].mean():.2%}")
    return df


def _off_hours_distribution() -> np.ndarray:
    """Probability distribution biased toward late night / early morning."""
    probs = np.ones(24)
    probs[0:6] = 4.0    # midnight to 6am: higher fraud
    probs[22:24] = 3.0  # 10pm to midnight
    probs[9:17] = 1.0   # business hours: lower fraud
    return probs / probs.sum()


def load_data(use_synthetic: bool = False, **kwargs) -> pd.DataFrame:
    """Main entry point: loads real data if available, falls back to synthetic."""
    if use_synthetic:
        return generate_synthetic_data(**kwargs)
    try:
        return load_kaggle_data(**kwargs)
    except FileNotFoundError:
        print("Kaggle data not found — generating synthetic data for demo.")
        return generate_synthetic_data(**kwargs)
