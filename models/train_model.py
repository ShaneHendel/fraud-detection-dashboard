"""
Model training pipeline for credit card fraud detection.
Trains an XGBoost classifier and saves model artifacts + metrics.

Usage:
    python models/train_model.py
    python models/train_model.py --synthetic   # use synthetic data for demo
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_data
from utils.feature_engineering import engineer_features, get_feature_columns

MODEL_DIR = Path(__file__).parent
ARTIFACTS_DIR = MODEL_DIR / "artifacts"


def train(use_synthetic: bool = False):
    """Full training pipeline: load → engineer → train → evaluate → save."""

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # --- Load & engineer ---
    print("=" * 60)
    print("FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)

    df = load_data(use_synthetic=use_synthetic)
    df = engineer_features(df)
    feature_cols = get_feature_columns()

    # Ensure all feature columns exist
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing features (will be dropped): {missing}")
    feature_cols = available

    X = df[feature_cols].fillna(0)
    y = df["is_fraud"]

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X):,} | Fraud: {y.sum():,} ({y.mean():.2%})")

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Train ---
    print("\nTraining XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="auc",
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- Evaluate ---
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*40}")
    print(f"ROC AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    # --- Cross-validation ---
    print("Running 5-fold CV...")
    cv_scores = cross_val_score(
        model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc", n_jobs=-1,
    )
    print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # --- Feature importance ---
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # --- Save artifacts ---
    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    metrics = {
        "roc_auc": float(auc),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "classification_report": report,
        "feature_importance": importance,
        "feature_columns": feature_cols,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test": float(y_test.mean()),
        "threshold": 0.5,
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save test set for dashboard use
    test_df = df.iloc[X_test.index].copy()
    test_df["fraud_probability"] = y_proba
    test_df["predicted_fraud"] = y_pred
    test_df.to_parquet(ARTIFACTS_DIR / "test_scored.parquet", index=False)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    print(f"  - model.pkl")
    print(f"  - metrics.json")
    print(f"  - test_scored.parquet")
    print("Done!")

    return model, metrics


if __name__ == "__main__":
    use_synthetic = "--synthetic" in sys.argv
    train(use_synthetic=use_synthetic)
