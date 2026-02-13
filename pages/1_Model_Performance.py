"""
📈 Model Performance — Monitoring dashboard page.
Tracks precision, recall, ROC AUC across simulated production batches.
"""

import json
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.monitoring import simulate_production_batches, get_curve_data, get_confusion_matrix_data
from utils.feature_engineering import get_feature_columns

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance Monitoring")

ARTIFACTS_DIR = Path("models/artifacts")

# --- Load artifacts ---
@st.cache_resource
def load_model():
    with open(ARTIFACTS_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_scored_data():
    return pd.read_parquet(ARTIFACTS_DIR / "test_scored.parquet")

@st.cache_data
def load_metrics():
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        return json.load(f)

model = load_model()
scored_df = load_scored_data()
metrics = load_metrics()
feature_cols = metrics["feature_columns"]

# --- Threshold slider ---
st.sidebar.markdown("### ⚙️ Settings")
threshold = st.sidebar.slider("Classification Threshold", 0.1, 0.95, 0.5, 0.05)

y_true = scored_df["is_fraud"].values
y_proba = scored_df["fraud_probability"].values
y_pred = (y_proba >= threshold).astype(int)

# --- ROC & PR Curves ---
st.markdown("### ROC & Precision-Recall Curves")
curve_data = get_curve_data(y_true, y_proba)

fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC Curve", "Precision-Recall Curve"))

fig.add_trace(
    go.Scatter(x=curve_data["roc"]["fpr"], y=curve_data["roc"]["tpr"],
               mode="lines", name=f'ROC (AUC={metrics["roc_auc"]:.3f})',
               line=dict(color="#0f172a", width=2.5)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
               line=dict(color="#cbd5e1", dash="dash")),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=curve_data["pr"]["recall"], y=curve_data["pr"]["precision"],
               mode="lines", name="PR Curve",
               line=dict(color="#dc2626", width=2.5)),
    row=1, col=2
)

fig.update_layout(height=400, font=dict(family="DM Sans"), plot_bgcolor="rgba(0,0,0,0)",
                  margin=dict(t=40, b=20))
fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
fig.update_xaxes(title_text="Recall", row=1, col=2)
fig.update_yaxes(title_text="Precision", row=1, col=2)
st.plotly_chart(fig, use_container_width=True)

# --- Confusion Matrix ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Confusion Matrix")
    cm = get_confusion_matrix_data(y_true, y_pred)
    cm_array = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])

    fig_cm = px.imshow(
        cm_array,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Legit", "Fraud"], y=["Legit", "Fraud"],
        text_auto=True,
        color_continuous_scale=["#f8fafc", "#0f172a"],
    )
    fig_cm.update_layout(height=350, font=dict(family="DM Sans"), margin=dict(t=10, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    st.markdown("### Metrics at Current Threshold")
    from sklearn.metrics import precision_score, recall_score, f1_score
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
    met_col1.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
    met_col2.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
    met_col3.metric("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
    met_col4.metric("Flagged", f"{y_pred.sum():,} / {len(y_pred):,}")

    # --- Score distribution ---
    st.markdown("### Score Distribution")
    score_df = pd.DataFrame({"probability": y_proba, "actual": y_true.astype(str)})
    score_df["actual"] = score_df["actual"].map({"0": "Legitimate", "1": "Fraud"})

    fig_dist = px.histogram(
        score_df, x="probability", color="actual", nbins=60, barmode="overlay",
        color_discrete_map={"Legitimate": "#94a3b8", "Fraud": "#dc2626"},
        opacity=0.7,
    )
    fig_dist.add_vline(x=threshold, line_dash="dash", line_color="#0f172a",
                       annotation_text=f"Threshold: {threshold}")
    fig_dist.update_layout(height=300, font=dict(family="DM Sans"),
                           plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10))
    st.plotly_chart(fig_dist, use_container_width=True)

# --- Batch performance over time ---
st.markdown("### Simulated Batch Performance Over Time")
st.caption("Splits test data into chronological batches to simulate production monitoring.")

batch_metrics = simulate_production_batches(scored_df, model, feature_cols, n_batches=10, threshold=threshold)

fig_batch = go.Figure()
for metric_name, color in [("precision", "#0f172a"), ("recall", "#dc2626"), ("f1", "#2563eb")]:
    fig_batch.add_trace(go.Scatter(
        x=batch_metrics["batch"], y=batch_metrics[metric_name],
        mode="lines+markers", name=metric_name.capitalize(),
        line=dict(color=color, width=2),
    ))

fig_batch.update_layout(
    height=350, font=dict(family="DM Sans"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis_title="Score", xaxis_title="Batch #",
    margin=dict(t=10, b=10),
)
st.plotly_chart(fig_batch, use_container_width=True)
