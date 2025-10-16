import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_insurance_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Insurance dataset not found at: {csv_path}")
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


def train_random_forest(
    X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer, photos_dir: str
) -> Tuple[RandomForestRegressor, float, float]:
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("rf", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RandomForest — MSE: {mse:.2f} | R^2: {r2:.4f}")

    # Actual vs Pred plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 5))
    ax_scatter.scatter(y_test, y_pred, s=16, alpha=0.6)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    ax_scatter.set_title("Insurance — Actual vs Predicted (RandomForest)")
    ax_scatter.set_xlabel("Actual charges")
    ax_scatter.set_ylabel("Predicted charges")
    fig_scatter.tight_layout()
    out_scatter = os.path.join(photos_dir, "insurance_actual_vs_pred_RandomForestRegressor.png")
    fig_scatter.savefig(out_scatter, dpi=150)
    plt.close(fig_scatter)
    print(f"Saved actual vs predicted to: {out_scatter}")

    # Feature importances
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = list(ohe.get_feature_names_out())
    num_feature_names = preprocessor.transformers_[0][2]
    feature_names = list(num_feature_names) + cat_feature_names

    rf: RandomForestRegressor = pipeline.named_steps["rf"]
    importances = rf.feature_importances_
    importances_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    importances_series.iloc[:20].plot(kind="barh", ax=ax_imp, color="#2ca02c")
    ax_imp.invert_yaxis()
    ax_imp.set_title("Insurance — Feature Importances (RandomForest)")
    ax_imp.set_xlabel("Importance")
    fig_imp.tight_layout()
    out_imp = os.path.join(photos_dir, "insurance_feature_importances_RandomForestRegressor.png")
    fig_imp.savefig(out_imp, dpi=150)
    plt.close(fig_imp)
    print(f"Saved feature importances to: {out_imp}")

    return rf, mse, r2


def save_correlation_heatmap(df: pd.DataFrame, photos_dir: str) -> None:
    # Only numeric columns
    corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Insurance — Correlation Heatmap")
    fig.tight_layout()
    out_path = os.path.join(photos_dir, "insurance_corr_heatmap.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved correlation heatmap to: {out_path}")


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, "data", "insurance", "insurance.csv")
    photos_dir = os.path.join(project_root, "photos")
    os.makedirs(photos_dir, exist_ok=True)

    df = load_insurance_dataset(data_path)

    target_column = "charges"
    feature_columns = [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region",
    ]

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]
    preprocessor = build_feature_pipeline(numeric_features, categorical_features)

    save_correlation_heatmap(df, photos_dir)

    train_random_forest(X, y, preprocessor, photos_dir)


if __name__ == "__main__":
    main()


