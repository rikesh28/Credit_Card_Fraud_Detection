# Credit Card Fraud Detection

End-to-end fraud detection system built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset — from exploratory analysis and model training to a deployed REST API and interactive dashboard.

**590K e-commerce transactions | 3.5% fraud rate | 400+ raw features**

**Live demo:** [API docs](https://credit-card-fraud-detection-api-lmas.onrender.com/docs) | [Dashboard](https://creditcardfrauddetectiondashboard.streamlit.app/)

## Results

| Model | Features | ROC-AUC | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 434 | 0.77 | 0.08 | 0.14 |
| Logistic Regression (balanced) | 434 | 0.79 | 0.67 | 0.17 |
| Random Forest | 434 | 0.87 | 0.70 | 0.27 |
| **XGBoost** | **434** | **0.90** | **0.74** | **0.31** |
| XGBoost (production) | 20 | 0.82 | 0.67 | 0.21 |

The production model uses only 20 features that are available at transaction time (no historical lookups needed). It trades ~8% ROC-AUC for 5x faster inference (<100ms vs ~500ms), which makes it usable for real-time scoring.

## Project structure

```
notebooks/                          # ML exploration and training
  1) EDA.ipynb                        Dataset exploration, missing values, fraud rates
  2) pattern_analysis.ipynb           Amount distributions, email/device/time patterns
  3) preprocessing.ipynb              Cleaning, 15 engineered features, time-based split
  4) baseline_model.ipynb             Logistic regression + random forest baselines
  5) advanced_models.ipynb            XGBoost, hyperparameter tuning, threshold optimization
  6) Production_model.ipynb           20-feature model for deployment

api/                                # FastAPI backend
  app/main.py                         API endpoints (predict, batch, health, model info)
  app/model.py                        Preprocessing + inference logic
  app/schemas.py                      Pydantic request/response models
  models/                             Trained XGBoost model + feature list
  test_api.py                         API tests
  requirements.txt                    API dependencies

dashboard/                          # Streamlit frontend
  app.py                              Main page
  config.py                           API URL config
  pages/                              Single prediction, batch, model comparison
  requirements.txt                    Dashboard dependencies
```

Run notebooks in order — each one loads artifacts from the previous step.

## Quick start

### Notebooks (training)

Built on Google Colab. To run locally:

1. Download the IEEE-CIS dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)
2. Place `train_transaction.csv` and `train_identity.csv` in a `data/` folder
3. Update file paths at the top of each notebook
4. `pip install -r requirements.txt`

### API

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Docs at `http://localhost:8000/docs`. See [api/README.md](api/README.md) for endpoint details.

### Dashboard

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

By default it connects to the deployed API on Render. Set `API_URL` env var to point to localhost if running the API locally.

## How it works

1. **EDA & feature engineering** — Explored 590K transactions, identified key fraud signals (product type, email domains, transaction amounts), engineered 15 new features, and used a time-based train/test split to avoid data leakage from random splitting.

2. **Model selection** — Compared logistic regression, random forest, and XGBoost. XGBoost with default hyperparameters gave the best results (0.90 AUC). Interestingly, the tuned model underperformed the default because I ran grid search on a 20% subsample to save time, and the optimal params on the subsample didn't generalize well.

3. **Production model** — The research model needs 434 features, many of which (V/C/D features) require historical database lookups that add ~500ms latency. I built a separate 20-feature model using only data available at transaction time. It drops to 0.82 AUC but runs in <100ms.

4. **API** — FastAPI service that loads the production model, applies the same preprocessing pipeline, and returns fraud probability + risk level. Supports single and batch predictions.

5. **Dashboard** — Streamlit app that provides a UI on top of the API. Users can test individual transactions, upload CSVs for batch analysis, and compare model performance.

## Deployment

- **API**: Render free tier ([live](https://credit-card-fraud-detection-api-lmas.onrender.com/docs)). Cold starts take 30-60s after 15 min of inactivity.
- **Dashboard**: Streamlit Community Cloud ([live](https://creditcardfrauddetectiondashboard.streamlit.app/))

## Known issues

- **Data leakage**: Label encoders are fit on the full dataset before the train/test split. For a proper pipeline, they should be fit only on training data. The impact is minor for label encoding with tree models, but it's not correct practice.
- **Hyperparameter tuning**: GridSearchCV on a 20% subsample optimizing for recall. The tuned model (0.88 AUC) actually did worse than the default (0.90 AUC), so I kept the default config.
- **Label encoding**: Used for nominal categoricals like card brand. Target encoding or native XGBoost categorical handling would be better approaches.
- **No cross-validation**: Single time-based split only. K-fold with time-series awareness would provide confidence intervals on the metrics.

## Tech stack

- **ML**: Python, NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **API**: FastAPI, Uvicorn, Pydantic
- **Dashboard**: Streamlit, Plotly
- **Deployment**: Render (API), Streamlit Cloud (dashboard)

## Contact

Rikesh Sapkota — [LinkedIn](https://www.linkedin.com/in/rikesh-sapkota-b0591a29a/) — rikeshsapkota123@gmail.com
