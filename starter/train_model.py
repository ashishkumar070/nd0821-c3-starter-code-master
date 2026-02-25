# Script to train machine learning model.
import pandas as pd
import joblib
from ml.model import compute_slice_metrics
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from pathlib import Path

# -----------------------------
# Load Data
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "census.csv"
data = pd.read_csv(DATA_PATH, skipinitialspace=True)

# Drop columns not available at inference time
data = data.drop(columns=["fnlgt", "education-num", "capital-gain", "capital-loss"])

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
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print("Model Metrics:")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Fbeta: {fbeta}")

# -----------------------------
# Save Model Artifacts
# -----------------------------
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODEL_DIR / "model.pkl")
joblib.dump(encoder, MODEL_DIR / "encoder.pkl")
joblib.dump(lb, MODEL_DIR / "lb.pkl")
print("Model and encoder saved successfully")

# -----------------------------
# Slice Metrics
# -----------------------------
slice_results = compute_slice_metrics(
    model,
    test,
    "education",
    encoder,
    lb,
)

with open("slice_output.txt", "w") as f:
    for line in slice_results:
        f.write(line + "\n")

print("Slice metrics saved to slice_output.txt")