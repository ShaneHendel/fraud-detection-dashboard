"""
Model monitoring and drift detection utilities.
Simulates production batch scoring and tracks performance over time.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report,
)


def compute_batch_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute comprehensive classification metrics for a scored batch."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else None,
        "n_transactions": len(y_true),
        "n_flagged": int(y_pred.sum()),
        "n_actual_fraud": int(y_true.sum()),
        "fraud_rate": float(y_true.mean()),
        "flag_rate": float(y_pred.mean()),
    }


def simulate_production_batches(df: pd.DataFrame, model, feature_cols: list,
                                 n_batches: int = 12, threshold: float = 0.5) -> pd.DataFrame:
    """
    Simulate monthly production batches and compute metrics over time.
    Used to populate the monitoring dashboard with realistic drift data.
    """
    df = df.copy().sort_values("trans_date_trans_time").reset_index(drop=True)
    batch_size = len(df) // n_batches
    records = []

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size if i < n_batches - 1 else len(df)
        batch = df.iloc[start:end]

        X_batch = batch[feature_cols].fillna(0)
        y_true = batch["is_fraud"].values

        y_proba = model.predict_proba(X_batch)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        metrics = compute_batch_metrics(y_true, y_pred, y_proba)
        metrics["batch"] = i + 1
        metrics["batch_start"] = batch["trans_date_trans_time"].min()
        metrics["batch_end"] = batch["trans_date_trans_time"].max()
        records.append(metrics)

    return pd.DataFrame(records)


def get_confusion_matrix_data(y_true, y_pred) -> dict:
    """Return confusion matrix components for visualization."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def get_curve_data(y_true, y_proba) -> dict:
    """Compute ROC and PR curve data for plotting."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    return {
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr": {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
    }
