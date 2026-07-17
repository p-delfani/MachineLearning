from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class Config:
    data_path: str = "customer_churn.csv"
    target_column: str = "target"
    artifact_path: str = "stacking_classifier.joblib"
    report_path: str = "stacking_report.json"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5


def load_data(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    return pd.read_csv(file_path)


def validate_data(df: pd.DataFrame, target_column: str) -> None:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    if df[target_column].isna().any():
        raise ValueError("Target column contains missing values.")

    if df[target_column].nunique() != 2:
        raise ValueError("This script expects binary classification.")


def get_feature_types(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_pipeline(numeric_features: list[str], categorical_features: list[str], random_state: int) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    selector = SelectFromModel(
        estimator=LinearSVC(C=0.05, penalty="l1", dual=False, random_state=random_state, max_iter=4000)
    )

    base_models = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=random_state,
            ),
        ),
        (
            "lr",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
    ]

    final_estimator = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=random_state,
    )

    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=final_estimator,
        stack_method="auto",
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("model", stacking),
        ]
    )
    return pipeline


def find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate_at_threshold(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": round(threshold, 6),
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "f1": round(f1_score(y_true, y_pred), 6),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def main() -> None:
    config = Config()

    df = load_data(config.data_path)
    validate_data(df, config.target_column)

    x = df.drop(columns=[config.target_column])
    y = df[config.target_column]

    numeric_features, categorical_features = get_feature_types(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    pipeline = build_pipeline(numeric_features, categorical_features, config.random_state)

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

    # Out-of-fold probabilities for threshold optimization.
    oof_prob = cross_val_predict(
        clone(pipeline),
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    best_threshold, best_cv_f1 = find_best_threshold(y_train, oof_prob)

    pipeline.fit(x_train, y_train)

    test_prob = pipeline.predict_proba(x_test)[:, 1]
    metrics = evaluate_at_threshold(y_test, test_prob, best_threshold)

    selector = pipeline.named_steps["selector"]
    support_mask = selector.get_support()

    transformed_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    selected_features = transformed_feature_names[support_mask].tolist()

    artifact = {
        "pipeline": pipeline,
        "threshold": best_threshold,
        "selected_features": selected_features,
        "target_column": config.target_column,
        "raw_feature_names": x.columns.tolist(),
    }

    report = {
        "config": asdict(config),
        "cv_best_f1": round(best_cv_f1, 6),
        "optimized_threshold": round(best_threshold, 6),
        "test_metrics": metrics,
        "selected_feature_count": len(selected_features),
        "selected_features_preview": selected_features[:30],
    }

    joblib.dump(artifact, config.artifact_path)
    Path(config.report_path).write_text(json.dumps(report, ensure_ascii
