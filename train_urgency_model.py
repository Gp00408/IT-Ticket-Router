import os

import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"

LABEL_COL = "urgency"
MODEL_PATH = os.path.join(MODELS_DIR, "urgency_model.joblib")
REPORT_PATH = os.path.join(MODELS_DIR, "urgency_model_report.txt")

# Urgency is ordinal (Low < Medium < High); keep a fixed class order for
# readable reports/confusion matrices instead of alphabetical sorting.
CLASS_ORDER = ["Low", "Medium", "High"]


def load_split():
    X_train = load_npz(os.path.join(PROCESSED_DIR, "X_train.npz"))
    X_test = load_npz(os.path.join(PROCESSED_DIR, "X_test.npz"))
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "tickets_train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "tickets_test.csv"))
    return X_train, X_test, train_df[LABEL_COL], test_df[LABEL_COL]


def train_model(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> str:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    labels = [c for c in CLASS_ORDER if c in model.classes_]
    report = classification_report(y_test, y_pred, labels=labels, digits=3)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    lines = [
        f"Accuracy: {acc:.3f}",
        "",
        "Classification report:",
        report,
        "Confusion matrix (rows=true, cols=predicted):",
        f"Labels: {labels}",
        str(matrix),
    ]
    return "\n".join(lines)


def main():
    X_train, X_test, y_train, y_test = load_split()

    model = train_model(X_train, y_train)
    summary = evaluate(model, X_test, y_test)
    print(summary)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
