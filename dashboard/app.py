import streamlit as st

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Fraud Detection System")
st.markdown("---")

with st.sidebar:
    st.title("Navigation")
    st.markdown(
        """
    **Production Model (XGBoost)**
    - ROC-AUC: 82%
    - Recall: 67%
    - Precision: 13%
    - Inference: <100ms
    """
    )

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "XGBoost")
col2.metric("ROC-AUC", "82%")
col3.metric("Recall", "67%")
col4.metric("Inference", "<100ms")

st.markdown("---")

st.markdown(
    """
### Pages

- **Single Prediction** — Enter transaction details, get a fraud score
- **Batch Prediction** — Upload a CSV, analyze all rows at once
- **Model Performance** — Compare research vs production models

### How it works

The dashboard sends transaction data to a
[FastAPI backend](https://credit-card-fraud-detection-api-lmas.onrender.com/docs)
running the production XGBoost model (20 features, trained on 590K transactions
from the IEEE-CIS dataset).

The first request may be slow (~60s) due to Render free-tier cold starts.
"""
)

st.markdown("---")
st.caption(
    "Built by Rikesh Sapkota · "
    "[GitHub](https://github.com/rikesh28/Credit_Card_Fraud_Detection) · "
    "rikeshsapkota123@gmail.com"
)
