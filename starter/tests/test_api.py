from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "Welcome" in r.json()["message"]
    

high_income_input = {
    "age": 42,
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

low_income_input = {
    "age": 25,
    "workclass": "Private",
    "education": "HS-grad",
    "marital-status": "Never-married",
    "occupation": "Other-service",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Male",
    "hours-per-week": 20,
    "native-country": "United-States"
}

def test_post_prediction_high():
    r = client.post("/inference", json=high_income_input)
    assert r.status_code == 200
    assert r.json()["prediction"] == ">50K"

def test_post_prediction_low():
    r = client.post("/inference", json=low_income_input)
    assert r.status_code == 200
    assert r.json()["prediction"] == "<=50K"