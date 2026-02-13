"""
🔍 Transaction Explorer — Drill into flagged transactions.
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Transaction Explorer", page_icon="🔍", layout="wide")
st.title("🔍 Transaction Explorer")

ARTIFACTS_DIR = Path("models/artifacts")

@st.cache_data
def load_scored_data():
    df = pd.read_parquet(ARTIFACTS_DIR / "test_scored.parquet")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df

df = load_scored_data()

# --- Sidebar filters ---
st.sidebar.markdown("### 🎯 Filters")

threshold = st.sidebar.slider("Risk Threshold", 0.1, 0.95, 0.5, 0.05)
df["risk_flag"] = (df["fraud_probability"] >= threshold).astype(int)

view_mode = st.sidebar.radio("Show", ["All Transactions", "Flagged Only", "Fraud Only"])
if view_mode == "Flagged Only":
    df = df[df["risk_flag"] == 1]
elif view_mode == "Fraud Only":
    df = df[df["is_fraud"] == 1]

# Amount range
amt_min, amt_max = st.sidebar.slider(
    "Amount Range ($)", 
    float(df["amt"].min()), float(min(df["amt"].max(), 5000)),
    (float(df["amt"].min()), float(min(df["amt"].max(), 5000)))
)
df = df[(df["amt"] >= amt_min) & (df["amt"] <= amt_max)]

# Category filter
if "category" in df.columns:
    categories = st.sidebar.multiselect("Categories", sorted(df["category"].unique()))
    if categories:
        df = df[df["category"].isin(categories)]

# --- Summary stats ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions", f"{len(df):,}")
col2.metric("Flagged", f"{df['risk_flag'].sum():,}")
col3.metric("Actual Fraud", f"{df['is_fraud'].sum():,}")
col4.metric("Avg Amount", f"${df['amt'].mean():,.2f}")

# --- Risk score scatter ---
st.markdown("### Transaction Risk Map")

sample_size = min(5000, len(df))
plot_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df

fig = px.scatter(
    plot_df,
    x="amt",
    y="fraud_probability",
    color="fraud_probability",
    color_continuous_scale=["#fee2e2", "#dc2626"],
    opacity=0.5,
    hover_data=["category", "state", "amt", "fraud_probability"],
    labels={
        "amt": "Transaction Amount ($)",
        "fraud_probability": "Fraud Probability",
        "color": "Fraud Probability",
    },
)
fig.add_hline(y=threshold, line_dash="dash", line_color="#0f172a",
              annotation_text=f"Threshold: {threshold}")
fig.update_layout(height=450, font=dict(family="DM Sans"), plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

# --- Fraud by category ---
if "category" in df.columns:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Fraud Rate by Category")
        cat_stats = df.groupby("category").agg(
            fraud_rate=("is_fraud", "mean"),
            flag_rate=("risk_flag", "mean"),
            count=("is_fraud", "count"),
        ).sort_values("fraud_rate", ascending=True)

        fig_cat = px.bar(
            cat_stats.reset_index(), x="fraud_rate", y="category", orientation="h",
            color="fraud_rate", color_continuous_scale=["#f1f5f9", "#dc2626"],
        )
        fig_cat.update_layout(height=400, font=dict(family="DM Sans"),
                              plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
                              margin=dict(t=10))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        st.markdown("### Fraud by Hour of Day")
        if "hour" in df.columns:
            hour_stats = df.groupby("hour").agg(
                fraud_rate=("is_fraud", "mean"),
                count=("is_fraud", "count"),
            ).reset_index()

            fig_hour = px.bar(
                hour_stats, x="hour", y="fraud_rate",
                color="fraud_rate", color_continuous_scale=["#f1f5f9", "#dc2626"],
            )
            fig_hour.update_layout(height=400, font=dict(family="DM Sans"),
                                   plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
                                   margin=dict(t=10))
            st.plotly_chart(fig_hour, use_container_width=True)

# --- Data table ---
st.markdown("### Transaction Details")
display_cols = [c for c in [
    "trans_date_trans_time", "category", "amt", "state", "city_pop",
    "fraud_probability", "risk_flag", "is_fraud",
    "distance_to_merchant", "amt_zscore",
] if c in df.columns]

st.dataframe(
    df[display_cols].sort_values("fraud_probability", ascending=False).head(200),
    use_container_width=True,
    height=400,
)
