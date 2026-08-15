import os

import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_split(label_col: str):
    X_train = load_npz(os.path.join(PROCESSED_DIR, "X_train.npz"))
    X_test = load_npz(os.path.join(PROCESSED_DIR, "X_test.npz"))
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "tickets_train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "tickets_test.csv"))
    return X_train, X_test, train_df[label_col], test_df[label_col]


def train_model(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def format_confusion_matrix(matrix, labels) -> str:
    width = max(len(label) for label in labels) + 2
    rows = [" " * width + "".join(label.rjust(width) for label in labels)]
    for label, row in zip(labels, matrix):
        rows.append(label.rjust(width) + "".join(str(v).rjust(width) for v in row))
    return "\n".join(rows)
