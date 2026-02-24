from fastapi.testclient import TestClient
from starter.main import app

client = TestClient(app)


def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "Welcome" in r.json()["message"]


sample_input = {
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


def test_post_prediction():
    r = client.post("/inference", json=sample_input)

    assert r.status_code == 200
    assert r.json()["prediction"] in [">50K", "<=50K"]


def test_post_prediction_type():
    r = client.post("/inference", json=sample_input)

    assert isinstance(r.json()["prediction"], str)