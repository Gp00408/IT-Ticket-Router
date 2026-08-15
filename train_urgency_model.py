import os

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model_utils import MODELS_DIR, format_confusion_matrix, load_split, train_model

LABEL_COL = "urgency"
MODEL_PATH = os.path.join(MODELS_DIR, "urgency_model.joblib")
REPORT_PATH = os.path.join(MODELS_DIR, "urgency_model_report.txt")

# Urgency is ordinal (Low < Medium < High); keep a fixed class order for
# readable reports/confusion matrices instead of alphabetical sorting.
CLASS_ORDER = ["Low", "Medium", "High"]


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
        "Confusion matrix (rows = true urgency, columns = predicted urgency):",
        format_confusion_matrix(matrix, labels),
    ]
    return "\n".join(lines)


def main():
    X_train, X_test, y_train, y_test = load_split(LABEL_COL)

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
