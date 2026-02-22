import pickle
import pandas as pd
import numpy as np
from pathlib import Path

COMMON_EMAIL_PROVIDERS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"}


class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.optimal_threshold = 0.6
        self.model_loaded = False

    def load_model(self):
        model_path = Path(__file__).resolve().parent.parent / "models" / "api_model_xgb.pkl"
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        features_path = Path(__file__).resolve().parent.parent / "models" / "api_feature_names.csv"
        features_df = pd.read_csv(features_path)
        self.feature_names = features_df["feature"].tolist()
        self.model_loaded = True

    def preprocess_input(self, transaction_data: dict) -> pd.DataFrame:
        df = pd.DataFrame([transaction_data])

        if "TransactionAmt" in df.columns:
            df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
            df["TransactionAmt_decimal"] = (
                df["TransactionAmt"] - df["TransactionAmt"].astype(int)
            )
            df["is_round_amount"] = (df["TransactionAmt"] % 10 == 0).astype(int)

        if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
            df["P_emaildomain"] = df["P_emaildomain"].fillna("unknown")
            df["R_emaildomain"] = df["R_emaildomain"].fillna("unknown")

            df["email_domain_match"] = (
                df["P_emaildomain"] == df["R_emaildomain"]
            ).astype(int)
            df["P_email_is_common"] = (
                df["P_emaildomain"].isin(COMMON_EMAIL_PROVIDERS).astype(int)
            )
            df["R_email_is_common"] = (
                df["R_emaildomain"].isin(COMMON_EMAIL_PROVIDERS).astype(int)
            )
            df["has_P_email"] = (df["P_emaildomain"] != "unknown").astype(int)
            df["has_R_email"] = (df["R_emaildomain"] != "unknown").astype(int)

        # Label-encode remaining categoricals
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna("unknown").astype("category").cat.codes

        # Fill missing numericals with -999 (sentinel for tree models)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(-999)

        # Ensure all expected features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = -999

        return df[self.feature_names]

    def predict(self, transaction_data: dict) -> dict:
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        X = self.preprocess_input(transaction_data)
        fraud_probability = float(self.model.predict_proba(X)[0][1])
        is_fraud = fraud_probability >= self.optimal_threshold

        if fraud_probability < 0.3:
            risk_level = "Low"
        elif fraud_probability < 0.6:
            risk_level = "Medium"
        elif fraud_probability < 0.8:
            risk_level = "High"
        else:
            risk_level = "Critical"

        if is_fraud:
            message = f"Transaction flagged as FRAUD (confidence: {fraud_probability*100:.1f}%)"
        else:
            message = f"Transaction appears legitimate (fraud risk: {fraud_probability*100:.1f}%)"

        return {
            "is_fraud": bool(is_fraud),
            "fraud_probability": fraud_probability,
            "risk_level": risk_level,
            "message": message,
        }


fraud_model = FraudDetectionModel()
