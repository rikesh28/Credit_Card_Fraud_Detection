# Fraud Detection API

REST API for real-time fraud detection using XGBoost. Built with FastAPI.

**Live:** https://credit-card-fraud-detection-api-lmas.onrender.com/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Single transaction prediction |
| POST | `/predict/batch` | CSV upload for batch predictions |
| GET | `/model/info` | Model metadata and metrics |

## Example

```bash
curl -X POST "https://credit-card-fraud-detection-api-lmas.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 150.50,
    "ProductCD": "W",
    "card1": 13926,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }'
```

Response:
```json
{
  "is_fraud": false,
  "fraud_probability": 0.23,
  "risk_level": "Low",
  "message": "Transaction appears legitimate (fraud risk: 23.0%)"
}
```

## Running locally

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs at `http://localhost:8000/docs`.

## Running tests

```bash
cd api
pytest test_api.py -v
```

## Model

XGBoost classifier using 20 real-time features (reduced from 434 in the research model) for <100ms inference. The trained model and feature list are in `models/`.

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.82 |
| Recall | 0.67 |
| Precision | 0.13 |
| F1 | 0.21 |

## Deployment

Deployed on Render free tier. First request after 15 min of inactivity takes 30-60s (cold start).
