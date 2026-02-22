import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from config import API_URL

st.set_page_config(page_title="Batch Prediction", page_icon="📊", layout="wide")

st.title("Batch Transaction Processing")
st.markdown("Upload a CSV file to analyze multiple transactions.")
st.markdown("---")

# Sample CSV download
sample_data = pd.DataFrame({
    "TransactionAmt": [50.0, 150.5, 950.0],
    "ProductCD": ["W", "W", "C"],
    "card1": [13926, 14587, 99999],
    "card2": [150.0, 200.0, 100.0],
    "card3": [150.0, 200.0, 100.0],
    "card4": ["visa", "mastercard", "discover"],
    "card5": [226.0, 226.0, 150.0],
    "card6": ["debit", "credit", "credit"],
    "addr1": [315.0, 325.0, 999.0],
    "addr2": [87.0, 87.0, 50.0],
    "P_emaildomain": ["gmail.com", "yahoo.com", "suspicious.ru"],
    "R_emaildomain": ["gmail.com", "yahoo.com", "another-sus.com"],
})

st.download_button(
    "Download Sample CSV",
    sample_data.to_csv(index=False),
    "sample_transactions.csv",
    "text/csv",
)

st.markdown("---")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} transactions")

    with st.expander("Preview Data"):
        st.dataframe(df.head(10))

    if st.button("Analyze All", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {len(df)} transactions..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                response = requests.post(f"{API_URL}/predict/batch", files=files, timeout=180)

                if response.status_code == 200:
                    result = response.json()

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total", result["total_transactions"])
                    col2.metric("Fraud", result["fraud_detected"])
                    col3.metric("Legitimate", result["total_transactions"] - result["fraud_detected"])
                    fraud_rate = result["fraud_detected"] / result["total_transactions"] * 100
                    col4.metric("Fraud Rate", f"{fraud_rate:.1f}%")

                    predictions_df = pd.DataFrame(result["predictions"])

                    col1, col2 = st.columns(2)
                    with col1:
                        risk_counts = predictions_df["risk_level"].value_counts()
                        fig = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            title="Risk Distribution",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        fig = px.histogram(
                            predictions_df,
                            x="fraud_probability",
                            color="is_fraud",
                            title="Probability Distribution",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    high_risk = predictions_df[
                        predictions_df["risk_level"].isin(["High", "Critical"])
                    ]
                    if len(high_risk) > 0:
                        st.subheader("High Risk Transactions")
                        st.dataframe(high_risk, use_container_width=True)
                    else:
                        st.info("No high-risk transactions found.")

                    st.download_button(
                        "Download Results",
                        predictions_df.to_csv(index=False),
                        "fraud_predictions.csv",
                        "text/csv",
                    )
                else:
                    st.error(f"Error {response.status_code}")
                    st.code(response.text)

            except Exception as e:
                st.error(f"Error: {e}")
