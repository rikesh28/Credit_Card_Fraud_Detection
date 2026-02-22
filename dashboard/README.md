# Fraud Detection Dashboard

Streamlit web interface for the fraud detection API. Supports single and batch transaction analysis.

**Live:** https://creditcardfrauddetectiondashboard.streamlit.app/

## Pages

- **Single Prediction** — Enter transaction details, get fraud probability and risk level
- **Batch Prediction** — Upload a CSV, get predictions for all rows, download results
- **Model Performance** — Compare production vs research model metrics

## Running locally

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Set `API_URL` environment variable to point to a different API instance (defaults to the Render deployment).

## Architecture

The dashboard calls the FastAPI backend (in `api/`) for all predictions. No model runs locally in the dashboard — all inference happens server-side.
