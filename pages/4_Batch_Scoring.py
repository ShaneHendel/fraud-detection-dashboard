"""
📦 Batch Scoring — Upload new transaction CSVs for scoring.
"""

import json
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.feature_engineering import engineer_features, get_feature_columns

st.set_page_config(page_title="Batch Scoring", page_icon="📦", layout="wide")
st.title("📦 Batch Scoring")
st.caption("Upload a CSV of new transactions to score against the trained model.")

ARTIFACTS_DIR = Path("models/artifacts")


@st.cache_resource
def load_model():
    with open(ARTIFACTS_DIR / "model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metrics():
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        return json.load(f)


model = load_model()
metrics = load_metrics()
feature_cols = metrics["feature_columns"]

# --- File upload ---
uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
    help="CSV should match Kaggle fraud detection schema (amt, category, lat, long, merch_lat, merch_long, etc.)"
)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.markdown(f"### Uploaded: {len(raw_df):,} transactions")

    with st.expander("📋 Raw Data Preview", expanded=False):
        st.dataframe(raw_df.head(20), use_container_width=True)

    # --- Feature engineering ---
    try:
        # Parse dates
        if "trans_date_trans_time" in raw_df.columns:
            raw_df["trans_date_trans_time"] = pd.to_datetime(
                raw_df["trans_date_trans_time"], format="%d-%m-%Y %H:%M", dayfirst=True
            )
        if "dob" in raw_df.columns:
            raw_df["dob"] = pd.to_datetime(raw_df["dob"], format="%d-%m-%Y", dayfirst=True)

        # Need is_fraud column for feature engineering (category_fraud_rate)
        if "is_fraud" not in raw_df.columns:
            raw_df["is_fraud"] = 0  # placeholder for scoring

        engineered_df = engineer_features(raw_df)

        # Score
        available_features = [c for c in feature_cols if c in engineered_df.columns]
        X_new = engineered_df[available_features].fillna(0)

        # Pad missing columns with zeros
        for col in feature_cols:
            if col not in X_new.columns:
                X_new[col] = 0
        X_new = X_new[feature_cols]

        probabilities = model.predict_proba(X_new)[:, 1]
        engineered_df["fraud_probability"] = probabilities

        # --- Threshold ---
        threshold = st.slider("Classification Threshold", 0.1, 0.95, 0.5, 0.05)
        engineered_df["risk_flag"] = (probabilities >= threshold).astype(int)

        # --- Results summary ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Scored", f"{len(engineered_df):,}")
        col2.metric("Flagged", f"{engineered_df['risk_flag'].sum():,}")
        col3.metric("Flag Rate", f"{engineered_df['risk_flag'].mean():.1%}")
        col4.metric("Avg Risk Score", f"{probabilities.mean():.3f}")

        # --- Score distribution ---
        st.markdown("### Score Distribution")
        fig = px.histogram(
            engineered_df, x="fraud_probability", nbins=50,
            color_discrete_sequence=["#0f172a"],
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="#dc2626",
                       annotation_text=f"Threshold: {threshold}")
        fig.update_layout(height=300, font=dict(family="DM Sans"),
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # --- Flagged transactions ---
        st.markdown("### Flagged Transactions")
        flagged = engineered_df[engineered_df["risk_flag"] == 1].sort_values(
            "fraud_probability", ascending=False
        )

        display_cols = [c for c in [
            "trans_date_trans_time", "category", "amt", "state",
            "fraud_probability", "risk_flag",
            "distance_to_merchant", "amt_zscore",
        ] if c in flagged.columns]

        st.dataframe(flagged[display_cols].head(100), use_container_width=True, height=400)

        # --- Download scored results ---
        csv_output = engineered_df[display_cols + ["fraud_probability", "risk_flag"]].drop_duplicates(
            subset=[c for c in display_cols if c != "fraud_probability"]
        ).to_csv(index=False)

        st.download_button(
            "⬇️ Download Scored Results",
            csv_output,
            "scored_transactions.csv",
            "text/csv",
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Ensure your CSV matches the expected schema. See README for column requirements.")

else:
    st.info(
        "Upload a CSV file to get started. The file should contain columns like: "
        "`trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`, `lat`, `long`, "
        "`merch_lat`, `merch_long`, etc."
    )

    st.markdown("### Expected Schema")
    st.code("""
Required columns:
  amt            - Transaction amount
  category       - Merchant category
  lat, long      - Cardholder location
  merch_lat, merch_long - Merchant location
  trans_date_trans_time  - Transaction timestamp
  cc_num         - Card number (for velocity features)
  dob            - Date of birth (for age feature)
    """)
