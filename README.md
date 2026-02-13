# 🛡️ Credit Card Fraud Detection & Monitoring Dashboard

An end-to-end ML fraud detection pipeline with a production-style Streamlit monitoring dashboard. Built to demonstrate real-world fraud detection workflows: feature engineering, model training, batch scoring, explainability, and operational monitoring.

## Business Context

Credit card fraud costs the financial industry billions annually. This project simulates a production fraud detection system that:
- **Trains** a classification model on transaction-level behavioral features
- **Scores** incoming transaction batches and flags high-risk activity
- **Monitors** model performance over time (accuracy drift, precision/recall, feature importance shifts)
- **Explains** individual predictions using SHAP for analyst review

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Raw Transactions│────▶│ Feature Engineer  │────▶│  Model Training     │
│  (Kaggle CSV)    │     │ (engineered cols) │     │  (XGBoost/LightGBM) │
└─────────────────┘     └──────────────────┘     └────────┬────────────┘
                                                          │
                         ┌──────────────────┐             │
                         │ Streamlit Dashboard│◀───────────┘
                         │  • Monitoring      │
                         │  • Explainability  │
                         │  • Batch Scoring   │
                         │  • Alerts          │
                         └──────────────────┘
```

## Project Structure

```
fraud-detection-dashboard/
├── app.py                     # Streamlit dashboard (main entry)
├── pages/
│   ├── 1_Model_Performance.py # Precision, recall, ROC over time
│   ├── 2_Transaction_Explorer.py  # Drill into flagged transactions
│   ├── 3_SHAP_Explainability.py   # SHAP-based prediction explanations
│   └── 4_Batch_Scoring.py    # Upload & score new transaction batches
├── models/
│   └── train_model.py         # Training pipeline
├── utils/
│   ├── feature_engineering.py # Feature creation & transformation
│   ├── data_loader.py         # Data ingestion & validation
│   └── monitoring.py          # Drift detection & alerting logic
├── data/
│   └── (place Kaggle CSVs here)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (generates model artifact + metrics)
python models/train_model.py

# Launch the dashboard
streamlit run app.py
```

## Dataset

Uses the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) dataset.
Place `fraudTrain.csv` and `fraudTest.csv` in the `data/` directory.

## Key Features Engineered

| Feature | Description |
|---------|-------------|
| `hour`, `day_of_week` | Temporal patterns in fraud timing |
| `amt_zscore` | Transaction amount deviation from cardholder norm |
| `distance_to_merchant` | Geo distance between cardholder and merchant |
| `tx_frequency_1h` | Transaction velocity (count in rolling 1-hour window) |
| `category_fraud_rate` | Historical fraud rate by merchant category |
| `amt_to_median_ratio` | Amount relative to cardholder's median spend |

## Tech Stack

- **ML**: scikit-learn, XGBoost, SHAP
- **Dashboard**: Streamlit
- **Data**: pandas, numpy
- **Visualization**: Plotly, matplotlib

## Author

Shane Hendel — [LinkedIn](https://www.linkedin.com/in/shanehendel) | [Virtual Resume](https://huggingface.co/spaces/shanehendel/virtualresume)
