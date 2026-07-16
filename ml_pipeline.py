from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Config:
    data_path: str = "data.csv"
    target_column: str = "target"
    model_output_path: str = "model.joblib"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5


def load_data(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y


def detect_column_types(x: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_features = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = x.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    return numeric_features, categorical_features


def build_pipeline(
    numeric_features: List[str], categorical_features: List[str]
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(x_test)

    print("Test Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Test F1 Score:", round(f1_score(y_test, y_pred, average="weighted"), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if len(y_test.unique()) == 2:
        y_proba = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print("ROC-AUC:", round(auc, 4))


def main() -> None:
    config = Config()

    df = load_data(config.data_path)
    x, y = split_features_target(df, config.target_column)

    numeric_features, categorical_features = detect_column_types(x)
    pipeline = build_pipeline(numeric_features, categorical_features)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    cv_scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=config.cv_folds,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    print("Cross-validation F1 scores:", [round(score, 4) for score in cv_scores])
    print("Mean CV F1:", round(cv_scores.mean(), 4))

    pipeline.fit(x_train, y_train)
    evaluate_model(pipeline, x_test, y_test)

    joblib.dump(pipeline, config.model_output_path)
    print(f"\nModel saved to: {config.model_output_path}")


if __name__ == "__main__":
    main()
