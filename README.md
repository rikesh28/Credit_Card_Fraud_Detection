# Credit Card Fraud Detection

Binary classification on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (590K transactions, 3.5% fraud rate, 400+ features). Includes the full ML pipeline, a FastAPI backend for real-time predictions, and a Streamlit dashboard.

**Live:** [API docs](https://credit-card-fraud-detection-api-lmas.onrender.com/docs) | [Dashboard](https://creditcardfrauddetectiondashboard.streamlit.app/)

## Results

| Model | Features | ROC-AUC | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 434 | 0.77 | 0.08 | 0.14 |
| Logistic Regression (balanced) | 434 | 0.79 | 0.67 | 0.17 |
| Random Forest | 434 | 0.87 | 0.70 | 0.27 |
| **XGBoost** | **434** | **0.90** | **0.74** | **0.31** |
| XGBoost (production, 20 features) | 20 | 0.82 | 0.67 | 0.21 |

Best research model gets 0.90 ROC-AUC. Production model trades ~8% AUC for 5x faster inference (<100ms) by dropping features that require historical lookups.

## Project structure

```
notebooks/                             ML exploration and training
  1) EDA.ipynb                           Fraud rate by product/card/device, missing values
  2) pattern_analysis.ipynb              Amount distributions, email/time patterns
  3) preprocessing.ipynb                 Feature engineering (15 new), time-based split
  4) baseline_model.ipynb                Logistic regression, random forest
  5) advanced_models.ipynb               XGBoost,hyperparameter tuning,threshold optimization
  6) Production_model.ipynb              20-feature model for deployment

api/                                   FastAPI backend
  app/main.py                            Endpoints: predict, batch, health, model info
  app/model.py                           Preprocessing and inference
  app/schemas.py                         Pydantic request/response schemas
  models/                                Trained XGBoost model + feature list
  test_api.py                            API tests (pytest)
  requirements.txt

dashboard/                             Streamlit frontend
  app.py                                 Main page
  config.py                              API URL config
  pages/                                 Single prediction, batch, model comparison
  requirements.txt
```

## Setup

### Notebooks

Built for **Google Colab** with data from Google Drive. To reproduce:

1. Download the IEEE-CIS dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)
2. Upload `train_transaction.csv` and `train_identity.csv` to your Google Drive
3. Update file paths at the top of each notebook
4. Run notebooks in order (1 through 6)

Local: `pip install -r requirements.txt`

### API

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Docs at `http://localhost:8000/docs`. See [api/README.md](api/README.md) for endpoint details and examples.

### Dashboard

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Connects to the deployed API by default. Set `API_URL` env var to use a local API instance.

## How it works

1. **EDA & feature engineering** = Explored fraud patterns across product types, card brands, email domains, and time of day. Engineered 15 new features and used a time-based train/test split to avoid leaking future data.

2. **Model selection** = Tried logistic regression, random forest, and XGBoost. XGBoost with default params gave the best results (0.90 AUC). The tuned version actually scored lower (0.88) because I ran GridSearchCV on a 20% subsample to save time, and those params didn't generalize.

3. **Production model** = Most of the 434 research features (V/C/D columns) need historical database lookups (~500ms). Built a 20-feature model using only transaction-time data. Drops to 0.82 AUC but runs in <100ms.

4. **API** = FastAPI service serving the production model. Handles preprocessing, returns fraud probability and risk level. Supports single and batch (CSV upload) predictions.

5. **Dashboard** = Streamlit UI on top of the API. Test individual transactions, upload CSVs for batch analysis, compare production vs research model metrics.

## Deployment

- **API**: Render free tier. Cold starts take 30-60s after 15 min idle.
- **Dashboard**: Streamlit Community Cloud.

## Known limitations

- Label encoding is used for nominal categoricals (e.g., card brand). Target encoding would be more principled but label encoding works adequately with tree models on this dataset.
- The hyperparameter-tuned XGBoost actually scored slightly lower than the default configuration (0.88 vs 0.90 ROC-AUC). This happened because tuning was done on a 20% subsample optimizing for recall rather than AUC. The default XGBoost is the stronger research model.
- Feature engineering and encoding is done before the train/test split in the preprocessing notebook. In a production pipeline you'd fit encoders only on training data.

## Contact

Rikesh Sapkota - [LinkedIn](https://www.linkedin.com/in/rikesh-sapkota-b0591a29a/) - rikeshsapkota123@gmail.com
