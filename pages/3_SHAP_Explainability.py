"""
🧠 SHAP Explainability — Understand individual fraud predictions.
"""

import json
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="SHAP Explainability", page_icon="🧠", layout="wide")
st.title("🧠 SHAP Explainability")
st.caption("Understand why the model flagged (or didn't flag) individual transactions.")

ARTIFACTS_DIR = Path("models/artifacts")


@st.cache_resource
def load_model():
    with open(ARTIFACTS_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_scored_data():
    df = pd.read_parquet(ARTIFACTS_DIR / "test_scored.parquet")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


@st.cache_data
def load_metrics():
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_resource
def get_shap_explainer(_model, X_sample):
    """Create SHAP TreeExplainer (cached)."""
    try:
        import shap
        explainer = shap.TreeExplainer(_model)
        return explainer
    except ImportError:
        st.error("SHAP not installed. Run: `pip install shap`")
        return None


model = load_model()
scored_df = load_scored_data()
metrics = load_metrics()
feature_cols = metrics["feature_columns"]

# --- Select transaction ---
st.sidebar.markdown("### 🎯 Transaction Selection")
selection_mode = st.sidebar.radio("Select by", ["Highest Risk", "Transaction Index", "Random"])

if selection_mode == "Highest Risk":
    n_top = st.sidebar.slider("Top N risky transactions", 1, 50, 10)
    candidates = scored_df.nlargest(n_top, "fraud_probability")
    selected_idx = st.sidebar.selectbox(
        "Choose transaction",
        candidates.index.tolist(),
        format_func=lambda x: f"#{x} — ${scored_df.loc[x, 'amt']:.2f} (p={scored_df.loc[x, 'fraud_probability']:.3f})"
    )
elif selection_mode == "Transaction Index":
    selected_idx = st.sidebar.number_input("Index", 0, len(scored_df) - 1, 0)
else:
    if st.sidebar.button("🎲 Random Transaction"):
        st.session_state["random_idx"] = np.random.randint(0, len(scored_df))
    selected_idx = st.session_state.get("random_idx", 0)

tx = scored_df.iloc[selected_idx] if selection_mode != "Transaction Index" else scored_df.loc[selected_idx]

# --- Transaction summary ---
st.markdown("### Selected Transaction")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Amount", f"${tx['amt']:,.2f}")
col2.metric("Fraud Probability", f"{tx['fraud_probability']:.3f}")
col3.metric("Actual Label", "🔴 Fraud" if tx["is_fraud"] == 1 else "🟢 Legit")
col4.metric("Category", tx.get("category", "N/A"))
col5.metric("State", tx.get("state", "N/A"))

# --- Compute SHAP values ---
X_tx = pd.DataFrame([tx[feature_cols]]).fillna(0)
X_background = scored_df[feature_cols].fillna(0).sample(min(500, len(scored_df)), random_state=42)

explainer = get_shap_explainer(model, X_background)

if explainer is not None:
    import shap

    shap_values = explainer.shap_values(X_tx)

    # For binary classification, shap_values might be a list [class_0, class_1]
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # Class 1 (fraud) SHAP values
    else:
        sv = shap_values[0]

    # --- Waterfall-style bar chart ---
    st.markdown("### Feature Contributions to Fraud Prediction")

    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_value": sv,
        "feature_value": X_tx.values[0],
    }).sort_values("shap_value", key=abs, ascending=False)

    colors = ["#dc2626" if v > 0 else "#2563eb" for v in shap_df["shap_value"]]

    fig = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in shap_df["shap_value"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<br>Value: %{customdata:.2f}",
        customdata=shap_df["feature_value"],
    ))
    fig.update_layout(
        height=450,
        font=dict(family="DM Sans"),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="SHAP Value (impact on fraud prediction)",
        yaxis_title="",
        margin=dict(l=0, t=10, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "> 🔴 **Red bars** push the prediction toward **fraud**. "
        "🔵 **Blue bars** push toward **legitimate**."
    )

    # --- Feature value context ---
    st.markdown("### Feature Values in Context")
    st.caption("How this transaction's features compare to the overall distribution.")

    context_col1, context_col2 = st.columns(2)

    top_features = shap_df.head(6)["feature"].tolist()
    for i, feat in enumerate(top_features):
        col = context_col1 if i % 2 == 0 else context_col2
        with col:
            feat_val = float(X_tx[feat].iloc[0])
            pop_mean = float(scored_df[feat].mean())
            pop_std = float(scored_df[feat].std()) or 1.0
            z = (feat_val - pop_mean) / pop_std
            st.markdown(
                f"**{feat}**: `{feat_val:.2f}` "
                f"(population mean: {pop_mean:.2f}, z-score: {z:+.1f})"
            )

    # --- Global SHAP summary (small sample) ---
    with st.expander("🌍 Global Feature Importance (SHAP)", expanded=False):
        st.caption("SHAP importance across a sample of transactions.")

        global_sample = scored_df[feature_cols].fillna(0).sample(min(200, len(scored_df)), random_state=42)
        global_shap = explainer.shap_values(global_sample)
        if isinstance(global_shap, list):
            global_shap = global_shap[1]

        mean_abs_shap = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": np.abs(global_shap).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=True)

        fig_global = go.Figure(go.Bar(
            x=mean_abs_shap["mean_abs_shap"],
            y=mean_abs_shap["feature"],
            orientation="h",
            marker_color="#0f172a",
        ))
        fig_global.update_layout(
            height=380, font=dict(family="DM Sans"),
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Mean |SHAP Value|",
            margin=dict(l=0, t=10),
        )
        st.plotly_chart(fig_global, use_container_width=True)

else:
    st.info("Install SHAP to enable explainability: `pip install shap`")
