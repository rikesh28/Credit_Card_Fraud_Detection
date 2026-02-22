# Credit Card Fraud Detection

Fraud classification on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset. 590K e-commerce transactions, 3.5% fraud rate, 400+ raw features.

## Results

| Model | Features | ROC-AUC | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 434 | 0.77 | 0.08 | 0.14 |
| Logistic Regression (balanced) | 434 | 0.79 | 0.67 | 0.17 |
| Random Forest | 434 | 0.87 | 0.70 | 0.27 |
| **XGBoost** | **434** | **0.90** | **0.74** | **0.31** |
| XGBoost (production) | 20 | 0.82 | 0.67 | 0.21 |

The production model uses only 20 features that are available at transaction time (no historical lookups). It trades ~8% ROC-AUC for 5x faster inference (<100ms vs ~500ms).

## Project structure

```
notebooks/
  1) EDA.ipynb                  - Dataset exploration, missing values, fraud rates by category
  2) pattern_analysis.ipynb     - Amount distributions, email/device/time patterns
  3) preprocessing.ipynb        - Cleaning, 15 engineered features, time-based split
  4) baseline_model.ipynb       - Logistic regression + random forest baselines
  5) advanced_models.ipynb      - XGBoost, hyperparameter tuning, threshold optimization
  6) Production_model.ipynb     - 20-feature model for deployment
```

Run notebooks in order. Each one loads artifacts from the previous step.

## Related repos

- [API (FastAPI)](https://github.com/rikesh28/Credit_Card_Fraud_Detection_API) - REST API serving the production model
- [Dashboard (Streamlit)](https://github.com/rikesh28/Credit_Card_Fraud_Detection_Dashboard) - Web interface for predictions

Live: [API docs](https://credit-card-fraud-detection-api-lmas.onrender.com/docs) | [Dashboard](https://creditcardfrauddetectiondashboard.streamlit.app/)

## Setup

Built on Google Colab. To reproduce locally:

1. Download the IEEE-CIS dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)
2. Place `train_transaction.csv` and `train_identity.csv` in a `data/` directory
3. Update file paths at the top of each notebook
4. `pip install -r requirements.txt`

## Known issues

- **Data leakage**: Label encoders are fit on the full dataset before the train/test split. For a production pipeline, encoders should be fit on training data only. The effect is minor for label encoding with tree models, but it's not best practice.
- **Hyperparameter tuning**: GridSearchCV was run on a 20% subsample optimizing for recall. The tuned model (0.88 AUC) actually underperformed the default (0.90 AUC). The default XGBoost config is the final research model.
- **Label encoding**: Used for nominal categoricals like card brand. Target encoding or leaving categories for native XGBoost handling would be better.
- **No cross-validation**: Single time-based split only. K-fold would give confidence intervals on metrics.

## Contact

Rikesh Sapkota - [LinkedIn](https://www.linkedin.com/in/rikesh-sapkota-b0591a29a/) - rikeshsapkota123@gmail.com
