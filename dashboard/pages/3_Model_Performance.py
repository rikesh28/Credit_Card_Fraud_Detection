import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.title("Model Performance")
st.markdown("Comparison of research and production fraud detection models.")

PRODUCTION = {"precision": 0.12, "recall": 0.67, "f1_score": 0.21, "roc_auc": 0.82}
RESEARCH = {"precision": 0.15, "recall": 0.75, "f1_score": 0.25, "roc_auc": 0.87}

model_view = st.radio(
    "Select model:",
    ["Production (Deployed)", "Research (Benchmark)", "Comparison"],
    horizontal=True,
)

st.markdown("---")

if model_view == "Production (Deployed)":
    metrics = PRODUCTION
    tn, fp, fn, tp = 94910, 19134, 1328, 2736

    st.markdown(
        "**20 features** | XGBoost | <100ms inference | "
        "472K training / 118K test samples"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{metrics['roc_auc']:.0%}")
    col2.metric("Precision", f"{metrics['precision']:.0%}")
    col3.metric("Recall", f"{metrics['recall']:.0%}")
    col4.metric("F1-Score", f"{metrics['f1_score']:.0%}")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        cm = np.array([[tn, fp], [fn, tp]])
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Pred Legitimate", "Pred Fraud"],
                y=["True Legitimate", "True Fraud"],
                colorscale="RdYlGn_r",
                text=[[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
                texttemplate="%{text}",
                textfont={"size": 18},
                showscale=False,
            )
        )
        fig.update_layout(title="Confusion Matrix", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"**TP:** {tp:,} frauds caught")
        st.markdown(f"**TN:** {tn:,} legitimate confirmed")
        st.markdown(f"**FP:** {fp:,} false alarms")
        st.markdown(f"**FN:** {fn:,} missed frauds")

    st.markdown("---")
    st.subheader("Feature Importance (Top 15)")
    features = pd.DataFrame({
        "Feature": [
            "TransactionAmt", "TransactionAmt_log", "card1", "addr1",
            "card2", "email_domain_match", "P_email_is_common", "addr2",
            "card5", "is_round_amount", "TransactionAmt_decimal", "card3",
            "has_P_email", "has_R_email", "R_email_is_common",
        ],
        "Importance": [
            0.18, 0.14, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05,
            0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02,
        ],
    })
    fig = px.bar(
        features, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Viridis",
    )
    fig.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif model_view == "Research (Benchmark)":
    metrics = RESEARCH
    tn, fp, fn, tp = 96883, 17211, 1001, 3063

    st.markdown(
        "**434 features** | XGBoost (tuned) | ~500ms inference | "
        "472K training / 118K test samples"
    )
    st.caption("Not deployed — requires historical data and a data warehouse.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{metrics['roc_auc']:.0%}")
    col2.metric("Precision", f"{metrics['precision']:.0%}")
    col3.metric("Recall", f"{metrics['recall']:.0%}")
    col4.metric("F1-Score", f"{metrics['f1_score']:.0%}")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        cm = np.array([[tn, fp], [fn, tp]])
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Pred Legitimate", "Pred Fraud"],
                y=["True Legitimate", "True Fraud"],
                colorscale="Blues",
                text=[[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
                texttemplate="%{text}",
                textfont={"size": 18},
                showscale=False,
            )
        )
        fig.update_layout(title="Confusion Matrix", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"**TP:** {tp:,} frauds caught")
        st.markdown(f"**TN:** {tn:,} legitimate confirmed")
        st.markdown(f"**FP:** {fp:,} false alarms")
        st.markdown(f"**FN:** {fn:,} missed frauds")

    st.markdown("---")
    st.subheader("Feature Categories")
    categories = pd.DataFrame({
        "Category": [
            "V-Features (Vesta)", "D-Features (Time)", "C-Features (Count)",
            "Metadata", "Card", "Email", "Engineered",
        ],
        "Count": [339, 15, 14, 12, 6, 2, 46],
        "Real-Time": ["No", "No", "No", "Yes", "Yes", "Yes", "Yes"],
    })
    fig = px.bar(
        categories, x="Count", y="Category", orientation="h",
        color="Real-Time",
        color_discrete_map={"Yes": "green", "No": "gray"},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Most features (V, D, C) require historical transaction data "
        "that isn't available at prediction time."
    )

else:  # Comparison
    st.subheader("Side-by-Side Metrics")

    comparison = pd.DataFrame({
        "Metric": ["ROC-AUC", "Precision", "Recall", "F1-Score"],
        "Research (434 features)": [
            f"{RESEARCH[k]:.0%}"
            for k in ["roc_auc", "precision", "recall", "f1_score"]
        ],
        "Production (20 features)": [
            f"{PRODUCTION[k]:.0%}"
            for k in ["roc_auc", "precision", "recall", "f1_score"]
        ],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC Gap", f"{(RESEARCH['roc_auc'] - PRODUCTION['roc_auc']):.0%}")
    col2.metric("Speed Gain", "5x", help="500ms -> <100ms")
    col3.metric("Feature Reduction", "95%", help="434 -> 20")

    st.markdown("---")

    categories = ["ROC-AUC", "Precision", "Recall", "F1-Score", "Speed"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[RESEARCH["roc_auc"], RESEARCH["precision"], RESEARCH["recall"],
           RESEARCH["f1_score"], 0.2],
        theta=categories, fill="toself", name="Research", line_color="blue",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[PRODUCTION["roc_auc"], PRODUCTION["precision"], PRODUCTION["recall"],
           PRODUCTION["f1_score"], 1.0],
        theta=categories, fill="toself", name="Production", line_color="green",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=450, title="Model Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    details = pd.DataFrame({
        "": ["Features", "Inference", "Infrastructure", "Real-Time"],
        "Research": ["434", "~500ms", "Data warehouse required", "No"],
        "Production": ["20", "<100ms", "API only", "Yes"],
    })
    st.dataframe(details, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Metrics from the IEEE-CIS Fraud Detection dataset (590K transactions). "
    "See the notebooks/ folder for training details."
)
