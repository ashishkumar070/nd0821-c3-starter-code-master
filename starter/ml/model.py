from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, fbeta_score
from ml.data import process_data



def train_model(X_train, y_train):
    """
    Train a machine learning model.

    Parameters
    ----------
    X_train : np.ndarray
        Training data
    y_train : np.ndarray
        Labels

    Returns
    -------
    model : LogisticRegression
        Trained machine learning model
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Compute precision, recall and fbeta score.

    Parameters
    ----------
    y : np.ndarray
        True labels
    preds : np.ndarray
        Predicted labels

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)

    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inference.

    Parameters
    ----------
    model : LogisticRegression
    X : np.ndarray

    Returns
    -------
    preds : np.ndarray
    """
    preds = model.predict(X)
    return preds


# ======================================================
# Slice Metrics (Udacity Requirement)
# ======================================================

def compute_slice_metrics(
    model,
    data,
    categorical_feature,
    encoder,
    lb,
):
    """
    Compute model performance on slices of a categorical feature.

    Example:
        education = Bachelors
        education = HS-grad
        education = Masters
    """


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

    results = []

    for category in data[categorical_feature].unique():

        slice_df = data[data[categorical_feature] == category]

        X, y, _, _ = process_data(
            slice_df,
            categorical_features=categorical_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = inference(model, X)

        precision, recall, fbeta = compute_model_metrics(
            y,
            preds,
        )

        results.append(
            (
                f"{categorical_feature}={category} | "
                f"Precision={precision:.3f}, "
                f"Recall={recall:.3f}, "
                f"Fbeta={fbeta:.3f}"
            )
        )

    return results