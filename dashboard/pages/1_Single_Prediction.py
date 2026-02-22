import streamlit as st
import requests
import plotly.graph_objects as go
from config import API_URL

st.set_page_config(page_title="Single Prediction", page_icon="🎯", layout="wide")

st.title("Single Transaction Prediction")
st.markdown("Enter transaction details to check for fraud.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Details")
    transaction_amt = st.number_input("Amount ($)", min_value=0.0, value=150.50, step=0.01)
    product_cd = st.selectbox("Product Type", ["W", "C", "H", "S", "R"])
    card1 = st.number_input("Card ID (card1)", min_value=0, value=13926)
    card4 = st.selectbox("Card Type", ["visa", "mastercard", "discover", "american express"])
    card6 = st.selectbox("Card Category", ["debit", "credit"])

with col2:
    st.subheader("Additional Info")
    card2 = st.number_input("Card ID 2", value=150.0)
    card3 = st.number_input("Card ID 3", value=150.0)
    card5 = st.number_input("Card ID 5", value=226.0)
    addr1 = st.number_input("Address 1", value=315.0)
    addr2 = st.number_input("Address 2", value=87.0)
    p_email = st.text_input("Purchaser Email Domain", value="gmail.com")
    r_email = st.text_input("Recipient Email Domain", value="gmail.com")

st.markdown("---")

if st.button("Check for Fraud", type="primary", use_container_width=True):
    payload = {
        "TransactionAmt": float(transaction_amt),
        "ProductCD": str(product_cd),
        "card1": int(card1),
        "card2": float(card2),
        "card3": float(card3),
        "card4": str(card4),
        "card5": float(card5),
        "card6": str(card6),
        "addr1": float(addr1),
        "addr2": float(addr2),
        "P_emaildomain": str(p_email),
        "R_emaildomain": str(r_email),
    }

    with st.spinner("Analyzing transaction (first request may take ~60s)..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()

                if result["is_fraud"]:
                    st.error(f"**FRAUD DETECTED** — {result['message']}")
                else:
                    st.success(f"**Legitimate** — {result['message']}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Fraud Probability", f"{result['fraud_probability']*100:.1f}%")
                col2.metric("Risk Level", result["risk_level"])
                col3.metric("Decision", "FRAUD" if result["is_fraud"] else "SAFE")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["fraud_probability"] * 100,
                    title={"text": "Fraud Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "red" if result["is_fraud"] else "green"},
                        "steps": [
                            {"range": [0, 30], "color": "lightgreen"},
                            {"range": [30, 60], "color": "yellow"},
                            {"range": [60, 100], "color": "lightcoral"},
                        ],
                    },
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            elif response.status_code == 422:
                st.error("Validation error")
                st.json(response.json())
            else:
                st.error(f"Error {response.status_code}")
                st.code(response.text)

        except requests.Timeout:
            st.warning("Request timed out. The API may be cold starting — try again in 30s.")
        except requests.ConnectionError:
            st.error(f"Cannot connect to API at {API_URL}")
        except Exception as e:
            st.error(f"Error: {e}")
