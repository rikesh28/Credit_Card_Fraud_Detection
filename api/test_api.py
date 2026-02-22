from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_TRANSACTION = {
    "TransactionAmt": 45.50,
    "ProductCD": "W",
    "card1": 13926,
    "card2": 150.0,
    "card3": 150.0,
    "card4": "visa",
    "card5": 226.0,
    "card6": "debit",
    "addr1": 315.0,
    "addr2": 87.0,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
}


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ml_model_loaded"] is True


def test_predict_returns_required_fields():
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "message" in data


def test_predict_probability_range():
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    data = response.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_risk_level_valid():
    response = client.post("/predict", json=SAMPLE_TRANSACTION)
    data = response.json()
    assert data["risk_level"] in {"Low", "Medium", "High", "Critical"}


def test_predict_missing_required_field():
    response = client.post("/predict", json={"ProductCD": "W"})
    assert response.status_code == 422


def test_predict_negative_amount():
    txn = {**SAMPLE_TRANSACTION, "TransactionAmt": -10.0}
    response = client.post("/predict", json=txn)
    assert response.status_code == 422


def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "XGBoost Classifier"
    assert "performance" in data
