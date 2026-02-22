# Credit Card Fraud Detection

Binary classification on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (590K transactions, 3.5% fraud rate, 400+ features).

## Results

| Model | Features | ROC-AUC | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 434 | 0.77 | 0.08 | 0.14 |
| Logistic Regression (balanced) | 434 | 0.79 | 0.67 | 0.17 |
| Random Forest | 434 | 0.87 | 0.70 | 0.27 |
| **XGBoost** | **434** | **0.90** | **0.74** | **0.31** |
| XGBoost (production, 20 features) | 20 | 0.82 | 0.67 | 0.21 |

Best research model gets 0.90 ROC-AUC. Production model trades ~8% AUC for 5x faster inference (<100ms) by dropping features that require historical lookups.

## Notebooks

Run in order:

1. **EDA** - Dataset exploration, fraud rate by product/card/device, missing value analysis
2. **Pattern Analysis** - Transaction amount distributions, email domain patterns, time-of-day effects, feature correlations
3. **Preprocessing** - Missing value handling, feature engineering (15 new features), label encoding, time-based train/test split
4. **Baseline Models** - Logistic regression (with/without class balancing), random forest. Demonstrates why class imbalance handling matters.
5. **Advanced Models** - XGBoost with hyperparameter tuning via GridSearchCV. Threshold optimization and business impact analysis.
6. **Production Model** - Reduced feature set (434 -> 20) using only real-time available features. This is the model behind the deployed API.

## Deployed

- API: https://credit-card-fraud-detection-api-lmas.onrender.com/docs
- Dashboard: https://creditcardfrauddetectiondashboard.streamlit.app/

## Setup

Notebooks are built for **Google Colab** with data loaded from Google Drive. To reproduce:

1. Download the IEEE-CIS dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)
2. Upload `train_transaction.csv` and `train_identity.csv` to your Google Drive
3. Update the file paths at the top of each notebook to match your Drive folder
4. Run notebooks in order (1 through 6)

Local setup:

```bash
pip install -r requirements.txt
```

## Known Limitations

- Label encoding is used for nominal categoricals (e.g., card brand). Target encoding would be more principled but label encoding works adequately with tree models on this dataset.
- The hyperparameter-tuned XGBoost actually scored slightly lower than the default configuration (0.88 vs 0.90 ROC-AUC). This happened because tuning was done on a 20% subsample optimizing for recall rather than AUC. The default XGBoost is the stronger research model.
- Feature engineering and encoding is done before the train/test split in the preprocessing notebook. In a production pipeline you'd fit encoders only on training data.

## Contact

Rikesh Sapkota - [LinkedIn](https://www.linkedin.com/in/rikesh-sapkota-b0591a29a/) - rikeshsapkota123@gmail.com
