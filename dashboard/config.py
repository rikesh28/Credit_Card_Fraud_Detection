import os

API_URL = os.getenv(
    "API_URL",
    "https://credit-card-fraud-detection-api-lmas.onrender.com",
).rstrip("/")
