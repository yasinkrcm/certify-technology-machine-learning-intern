import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def load_titanic_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Titanic dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def build_feature_pipeline(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def train_model(
    df: pd.DataFrame, photos_dir: str
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]:
    target_column = "Survived"
    feature_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]

    # Drop rows missing target
    df = df.dropna(subset=[target_column]).copy()

    X = df[feature_columns]
    y = df[target_column].astype(int)

    numeric_features = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
    categorical_features = ["Sex", "Embarked"]

    preprocessor = build_feature_pipeline(numeric_features, categorical_features)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("rf", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("Titanic - Confusion Matrix")
    fig_cm.tight_layout()
    cm_path = os.path.join(photos_dir, "titanic_confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)
    print(f"Saved confusion matrix to: {cm_path}")

    # Target distribution plot
    fig_td, ax_td = plt.subplots(figsize=(5, 4))
    y.value_counts().sort_index().plot(kind="bar", ax=ax_td, color=["#1f77b4", "#ff7f0e"])
    ax_td.set_title("Titanic - Target Distribution (Survived)")
    ax_td.set_xlabel("Survived")
    ax_td.set_ylabel("Count")
    fig_td.tight_layout()
    td_path = os.path.join(photos_dir, "titanic_target_distribution.png")
    fig_td.savefig(td_path, dpi=150)
    plt.close(fig_td)
    print(f"Saved target distribution to: {td_path}")

    # Feature importances from the trained RF on transformed features
    # Extract one-hot feature names to map importances
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = numeric_features + cat_feature_names

    rf: RandomForestClassifier = pipeline.named_steps["rf"]
    importances = rf.feature_importances_

    importances_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top_k = 20 if len(importances_series) > 20 else len(importances_series)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    importances_series.iloc[:top_k].plot(kind="barh", ax=ax_imp, color="#2ca02c")
    ax_imp.invert_yaxis()
    ax_imp.set_title("Titanic - Feature Importances (RandomForest)")
    ax_imp.set_xlabel("Importance")
    fig_imp.tight_layout()
    imp_path = os.path.join(photos_dir, "titanic_feature_importances.png")
    fig_imp.savefig(imp_path, dpi=150)
    plt.close(fig_imp)
    print(f"Saved feature importances to: {imp_path}")

    return rf, X_train, y_train, X_test, y_test, feature_names


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, "data", "titanic", "train.csv")
    photos_dir = os.path.join(project_root, "photos")
    os.makedirs(photos_dir, exist_ok=True)

    df = load_titanic_dataset(data_path)
    train_model(df, photos_dir)


if __name__ == "__main__":
    main()


