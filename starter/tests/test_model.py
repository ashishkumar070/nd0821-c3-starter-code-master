import numpy as np
from starter.model import train_model, inference, compute_model_metrics


def test_train_model():
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)

    model = train_model(X, y)

    assert model is not None


def test_inference():
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(y)


def test_compute_metrics():
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)