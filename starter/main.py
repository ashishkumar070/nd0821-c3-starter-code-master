from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

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
        json_schema_extra = {   # ✅ Pydantic v2 fix
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
    return {"message": "Welcome to the Census Income Prediction API"}


# -----------------------------
# POST Endpoint
# -----------------------------
@app.post("/inference")
def predict(data: CensusData):

    try:
        # Convert input
        input_df = pd.DataFrame([data.model_dump(by_alias=True)])

        # ---- ADD MISSING NUMERIC COLUMNS ----
        input_df["fnlgt"] = 0
        input_df["education-num"] = 0
        input_df["capital-gain"] = 0
        input_df["capital-loss"] = 0

        # Dummy label for encoder consistency
        input_df["salary"] = "<=50K"

        categorical_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        X, _, _, _ = process_data(
            input_df,
            categorical_features=categorical_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = inference(model, X)

        prediction = lb.inverse_transform(preds)[0]

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}