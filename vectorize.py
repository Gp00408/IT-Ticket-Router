import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz

from db import get_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_tickets(engine) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT ticket_text, category, urgency FROM tickets ORDER BY ticket_id", engine
    )


def main():
    engine = get_engine()
    df = load_tickets(engine)

    strata = df["category"] + "_" + df["urgency"]
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=strata
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    X_train = vectorizer.fit_transform(train_df["ticket_text"])
    X_test = vectorizer.transform(test_df["ticket_text"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    save_npz(os.path.join(PROCESSED_DIR, "X_train.npz"), X_train)
    save_npz(os.path.join(PROCESSED_DIR, "X_test.npz"), X_test)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "tickets_train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "tickets_test.csv"), index=False)

    print(f"Vectorized {X_train.shape[0]} train and {X_test.shape[0]} test tickets.")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


if __name__ == "__main__":
    main()
