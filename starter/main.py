from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

from ml.data import process_data
from ml.model import inference

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Census Income Prediction API"
)

# -----------------------------
# Load Model Artifacts
# -----------------------------
MODEL_PATH = "models/model.pkl"
ENCODER_PATH = "models/encoder.pkl"
LB_PATH = "models/lb.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
lb = joblib.load(LB_PATH)


# -----------------------------
# Pydantic Input Schema
# -----------------------------
class CensusData(BaseModel):

    age: int
    workclass: str
    education: str
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        populate_by_name = True
        schema_extra = {
            "example": {
                "age": 37,
                "workclass": "Private",
                "education": "Bachelors",
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


# -----------------------------
# GET Endpoint
# -----------------------------
@app.get("/")
def welcome():
    """
    Root endpoint returning welcome message
    """
    return {"message": "Welcome to the Census Income Prediction API"}


# -----------------------------
# POST Endpoint
# -----------------------------
@app.post("/inference")
def predict(data: CensusData):
    """
    Perform model inference
    """

    # Convert request to dataframe
    input_df = pd.DataFrame([data.dict(by_alias=True)])

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    # Process data (inference mode)
    X, _, _, _ = process_data(
        input_df,
        categorical_features=categorical_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Prediction
    preds = inference(model, X)

    prediction = lb.inverse_transform(preds)[0]

    return {"prediction": prediction}