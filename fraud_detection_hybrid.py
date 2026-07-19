from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


@dataclass
class Config:
    data_path: str = "transactions.csv"
    target_column: str = "is_fraud"
    artifact_path: str = "fraud_detector.joblib"
    report_path: str = "fraud_report.json"
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5


def load_data(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    return pd.read_csv(file_path)


def validate_data(df: pd.DataFrame, target_column: str) -> None:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    if df[target_column].nunique() != 2:
        raise ValueError("Target must be binary.")


def get_feature_types(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ])


def find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    idx = int(np.argmax(f1s))
    return float(thresholds[idx]), float(f1s[idx])


def main() -> None:
    config = Config()

    df = load_data(config.data_path)
    validate_data(df, config.target_column)

    x = df.drop(columns=[config.target_column])
    y = df[config.target_column].astype(int)

    numeric, categorical = get_feature_types(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    preprocessor = build_preprocessor(numeric, categorical)

    # Step 1: anomaly scores
    iso_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("iso", IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=config.random_state,
            n_jobs=-1,
        )),
    ])

    iso_pipeline.fit(x_train)
    anomaly_score_train = -iso_pipeline.named_steps["iso"].score_samples(
        iso_pipeline.named_steps["preprocessor"].transform(x_train)
    )
    anomaly_score_test = -iso_pipeline.named_steps["iso"].score_samples(
        iso_pipeline.named_steps["preprocessor"].transform(x_test)
    )

    # Step 2: supervised classifier with anomaly score as extra feature
    x_train_ext = x_train.copy()
    x_test_ext = x_test.copy()
    x_train_ext["anomaly_score"] = anomaly_score_train
    x_test_ext["anomaly_score"] = anomaly_score_test

    numeric2, categorical2 = get_feature_types(x_train_ext)
    preprocessor2 = build_preprocessor(numeric2, categorical2)

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        reg_alpha=0.0,
        scale_pos_weight=((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=config.random_state,
        n_jobs=-1,
    )

    clf = Pipeline([
        ("preprocessor", preprocessor2),
        ("model", CalibratedClassifierCV(xgb, cv=3, method="sigmoid")),
    ])

    clf.fit(x_train_ext, y_train)

    y_prob = clf.predict_proba(x_test_ext)[:, 1]
    threshold, best_f1 = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float((y_pred == y_test).mean()),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "best_f1": best_f1,
    }

    artifact = {
        "anomaly_detector": iso_pipeline,
        "classifier": clf,
        "threshold": threshold,
        "feature_columns": x.columns.tolist(),
        "target_column": config.target_column,
    }

    report = {
        "config": asdict(config),
        "metrics": metrics,
    }

    joblib.dump(artifact, config.artifact_path)
    Path(config.report_path).write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print("Threshold:", round(threshold, 4))
    print("F1:", round(metrics["f1"], 4))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))
    print("PR-AUC:", round(metrics["pr_auc"], 4))
    print(f"Saved: {config.artifact_path}")
    print(f"Saved: {config.report_path}")


if __name__ == "__main__":
    main()
