import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def load_sms_dataset(tsv_path: str) -> pd.DataFrame:
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"SMS dataset not found at: {tsv_path}")

    # Try to read with header, then fallback to no header
    try:
        df = pd.read_csv(tsv_path, sep="\t")
        # Normalize column names if needed
        cols = [c.lower() for c in df.columns]
        if "label" in cols and ("message" in cols or "text" in cols):
            df.columns = cols
        else:
            # Fallback to default names
            df = pd.read_csv(tsv_path, sep="\t", header=None, names=["label", "text"])
    except Exception:
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["label", "text"])

    if "text" not in df.columns:
        if "message" in df.columns:
            df.rename(columns={"message": "text"}, inplace=True)
        else:
            raise ValueError("Could not identify text column in SMS dataset")

    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    return df


def train_model(df: pd.DataFrame, photos_dir: str) -> Tuple[Pipeline, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("nb", MultinomialNB()),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Spam - Confusion Matrix (TF-IDF + MultinomialNB)")
    fig.tight_layout()
    out_path = os.path.join(photos_dir, "spam_confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to: {out_path}")

    return pipeline, acc


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, "data", "spam", "sms.tsv")
    photos_dir = os.path.join(project_root, "photos")
    os.makedirs(photos_dir, exist_ok=True)

    df = load_sms_dataset(data_path)
    train_model(df, photos_dir)


if __name__ == "__main__":
    main()


