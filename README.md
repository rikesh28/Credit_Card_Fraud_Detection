# Fraud Detection - ML Notebooks (End to End ML Project)
This is where I built and trained my fraud detection models. Everything from exploring the data to training the final model.

## What's Inside
I worked through this project step by step over several weeks:

**Week 1-2: Understanding the Data**
- `1) EDA.ipynb` - First time seeing 590K transactions. Learned that only 3.5% are fraud.
- `2) pattern_analysis.ipynb` - Dug deeper into patterns. Found some interesting things about transaction amounts and email domains.

**Week 3: Cleaning & Features**
- `3) preprocessing.ipynb` - Dealt with missing values (lots of them!). Created new features like log-transformed amounts and email domain matching.

**Week 4-5: Building Models**
- `4) baseline_model.ipynb` - Started simple with Logistic Regression.
Then did Random forest model. Also, I learned about class imbalance issue.
- `5) advanced_models.ipynb` - Built my best model with XGBoost. Got 88% ROC-AUC using 434 features.

**Week 5: Making It Production-Ready**
- `6) Production_model.ipynb` - This was the hard part. Realized 434 features won't work in real-time, so I cut it down to 21 features. Lost some accuracy (82% ROC-AUC) but gained speed.

## The Dataset
I used the IEEE-CIS Fraud Detection dataset from Kaggle:
- 590,291 transactions
- 3.5% fraud rate (pretty imbalanced!)
- 400+ features to work with

## What I Learned

**The Hard Way:**
- Class imbalance is real. My first model just predicted "not fraud" for everything and got 96% accuracy. Useless.
- More features ≠ better in production. My 434-feature model was accurate but way too slow.
- Accuracy is a terrible metric for fraud detection. Precision and recall matter way more.

**The Cool Stuff:**
- Feature engineering makes a huge difference. Simple things like "is this a round dollar amount" actually help.
- XGBoost is really good at this stuff compared to simpler models.
- You can get 90% of the performance with way fewer features if you pick the right ones.

## Key Results

**Research Model (434 features):**
- ROC-AUC: 88%
- Recall: 75%
- Problem: Takes 500ms per prediction, needs data warehouse

**Production Model (21 features):**
- ROC-AUC: 82%
- Recall: 67%
- Win: Under 100ms, works in real-time

I accepted 6% less accuracy to make it actually deployable. Turns out that's how it works in the real world.

## Running the Notebooks

You'll need:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

Then just:
```bash
jupyter notebook
```

Start with `1) EDA.ipynb` and work your way through.

## What's Next

The models from notebook 06 are deployed:
- API: [https://credit-card-fraud-detection-api-lmas.onrender.com/docs]
- Dashboard: [https://creditcardfrauddetectiondashboard.streamlit.app/]

Check out those repos to see how I turned this into a working system.

## Note

These are learning notebooks - they're messy in places because I was figuring things out as I went. That's the point. Real ML work is iterative and you learn by trying stuff.

---
## Contact
Built by Rikesh Sapkota

- LinkedIn: [https://www.linkedin.com/in/rikesh-sapkota-b0591a29a/]
- Email: rikeshsapkota123@gmail.com



