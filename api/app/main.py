from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import TransactionInput, PredictionResponse, HealthResponse
from app.model import fraud_model
import pandas as pd
from io import StringIO


@asynccontextmanager
async def lifespan(app: FastAPI):
    fraud_model.load_model()
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for e-commerce transactions using XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "ml_model_loaded": fraud_model.model_loaded,
        "ml_model_type": "XGBoost (Production)",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionInput):
    try:
        transaction_dict = transaction.model_dump()
        return fraud_model.predict(transaction_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        predictions = []
        for idx, row in df.iterrows():
            result = fraud_model.predict(row.to_dict())
            predictions.append({"transaction_id": idx, **result})

        return {
            "total_transactions": len(predictions),
            "fraud_detected": sum(1 for p in predictions if p["is_fraud"]),
            "predictions": predictions,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction error: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    return {
        "model_type": "XGBoost Classifier",
        "version": "2.0",
        "features_count": len(fraud_model.feature_names)
        if fraud_model.feature_names
        else "Unknown",
        "optimal_threshold": fraud_model.optimal_threshold,
        "performance": {
            "roc_auc": 0.82,
            "precision": 0.13,
            "recall": 0.67,
            "f1_score": 0.21,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
