# Script to train machine learning model.

import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


# -----------------------------
# Load Data
# -----------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "census.csv"

data = pd.read_csv(DATA_PATH, skipinitialspace=True)

# -----------------------------
# Train/Test Split
# -----------------------------
train, test = train_test_split(data, test_size=0.20)


# -----------------------------
# Categorical Features
# -----------------------------
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# -----------------------------
# Process Training Data
# -----------------------------
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)


# -----------------------------
# Process Test Data
# -----------------------------
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)


# -----------------------------
# Train Model
# -----------------------------
model = train_model(X_train, y_train)


# -----------------------------
# Evaluate Model
# -----------------------------
preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(
    y_test, preds
)

print("Model Metrics:")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Fbeta: {fbeta}")


# -----------------------------
# Save Model Artifacts
# -----------------------------
os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/model.pkl")
joblib.dump(encoder, "../models/encoder.pkl")
joblib.dump(lb, "../models/lb.pkl")

print("Model and encoder saved successfully")