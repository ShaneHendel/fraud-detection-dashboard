"""
ðŸ›¡ï¸ Credit Card Fraud Detection & Monitoring Dashboard
Main entry point for the Streamlit application.
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="ðŸ›¡ï¸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        font-family: 'DM Sans', sans-serif;
    }
    code, .stCode {
        font-family: 'JetBrains Mono', monospace;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0;
        line-height: 1.1;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #64748b;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #334155;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

ARTIFACTS_DIR = Path("models/artifacts")

# --- Rename root page label in default sidebar nav ---
components.html(
    """
    <script>
    const renameNav = () => {
      const links = window.parent.document.querySelectorAll('a[data-testid="stSidebarNavLink"]');
      const labelMap = {
        'app': '🛡️ Fraud Detection Dashboard',
        'model performance': '📈 Model Performance',
        'transaction explorer': '🔍 Transaction Explorer',
        'shap explainability': '🧠 SHAP Explainability',
        'batch scoring': '📦 Batch Scoring',
      };
      links.forEach((link) => {
        const label = link.querySelector('span');
        if (!label) return;
        const rawText = label.textContent.trim().toLowerCase();
        const normalizedText = rawText.replace(/^\\d+[\\s._-]*/, '');
        if (labelMap[normalizedText]) {
          label.textContent = labelMap[normalizedText];
        }
      });
    };
    renameNav();
    setTimeout(renameNav, 300);
    </script>
    """,
    height=0,
    width=0,
)

# --- Header ---
st.title("\U0001F6E1\uFE0F Fraud Detection Dashboard")
st.markdown(
    '<p class="hero-subtitle">'
    'End-to-end ML pipeline for credit card fraud detection â€” '
    'model training, batch scoring, performance monitoring, and SHAP explainability.'
    '</p>',
    unsafe_allow_html=True,
)

# --- Check for model artifacts ---
model_exists = (ARTIFACTS_DIR / "model.pkl").exists()

if not model_exists:
    st.warning(
        "âš ï¸ **No trained model found.** Run the training pipeline first:\n\n"
        "```bash\n"
        "python models/train_model.py            # with Kaggle data in data/\n"
        "python models/train_model.py --synthetic # with generated demo data\n"
        "```"
    )
    st.stop()

# --- Load metrics ---
import json
with open(ARTIFACTS_DIR / "metrics.json") as f:
    metrics = json.load(f)

# --- Key metrics row ---
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['roc_auc']:.3f}</div>
        <div class="metric-label">ROC AUC</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    report = metrics["classification_report"]
    prec = report.get("1", report.get("1.0", {})).get("precision", 0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{prec:.1%}</div>
        <div class="metric-label">Precision (Fraud)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    rec = report.get("1", report.get("1.0", {})).get("recall", 0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{rec:.1%}</div>
        <div class="metric-label">Recall (Fraud)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['n_train'] + metrics['n_test']:,}</div>
        <div class="metric-label">Total Transactions</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['fraud_rate_test']:.2%}</div>
        <div class="metric-label">Fraud Rate</div>
    </div>
    """, unsafe_allow_html=True)

# --- Feature Importance ---
st.markdown('<p class="section-header">Top Feature Importances</p>', unsafe_allow_html=True)

import plotly.express as px
import pandas as pd

fi = metrics["feature_importance"]
fi_df = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())})
fi_df = fi_df.sort_values("importance", ascending=True).tail(10)

fig = px.bar(
    fi_df, x="importance", y="feature", orientation="h",
    color="importance",
    color_continuous_scale=["#cbd5e1", "#0f172a"],
)
fig.update_layout(
    height=380,
    margin=dict(l=0, r=20, t=10, b=0),
    showlegend=False,
    coloraxis_showscale=False,
    yaxis_title="",
    xaxis_title="Importance Score",
    font=dict(family="DM Sans"),
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

# --- Navigation ---
st.markdown('<p class="section-header">\U0001F9ED Dashboard Pages</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**📈 Model Performance**\n\nPrecision, recall, ROC/PR curves, and confusion matrix.")
with col2:
    st.markdown("**🔍 Transaction Explorer**\n\nDrill into flagged transactions with filtering and detail views.")
with col3:
    st.markdown("**🧠 SHAP Explainability**\n\nUnderstand why individual transactions were flagged.")
with col4:
    st.markdown("**📦 Batch Scoring**\n\nUpload new transaction CSVs for scoring and monitoring.")

st.markdown("---")
st.caption("Built by Shane Hendel · [LinkedIn](https://www.linkedin.com/in/shanehendel)")

