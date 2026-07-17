from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


@dataclass
class Config:
    data_path: str = "customer_data.csv"
    target_column: str = "target"
    model_output_path: str = "advanced_catboost_model.joblib"
    metadata_output_path: str = "advanced_catboost_metadata.json"
    test_size: float = 0.15
    valid_size: float = 0.15
    random_state: int = 42

    iterations: int = 2000
    learning_rate: float = 0.03
    depth: int = 8
    l2_leaf_reg: float = 5.0
    eval_metric: str = "F1"
    loss_function: str = "Logloss"


def load_data(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    return pd.read_csv(file_path)


def validate_dataframe(df: pd.DataFrame, target_column: str) -> None:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    if df[target_column].isna().any():
        raise ValueError("Target column contains missing values.")

    if df[target_column].nunique() != 2:
        raise ValueError("This script expects a binary classification target.")


def split_data(
    df: pd.DataFrame, target_column: str, test_size: float, valid_size: float, random_state: int
):
    x = df.drop(columns=[target_column])
    y = df[target_column]

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    valid_ratio_adjusted = valid_size / (1.0 - test_size)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full,
        y_train_full,
        test_size=valid_ratio_adjusted,
        random_state=random_state,
        stratify=y_train_full,
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def detect_categorical_columns(x: pd.DataFrame) -> list[str]:
    return x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": round(threshold, 6),
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "f1": round(f1_score(y_true, y_pred), 6),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    return metrics


def build_model(config: Config, scale_pos_weight: float) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        loss_function=config.loss_function,
        eval_metric=config.eval_metric,
        random_seed=config.random_state,
        verbose=100,
        early_stopping_rounds=150,
        auto_class_weights=None,
        scale_pos_weight=scale_pos_weight,
    )


def main() -> None:
    config = Config()

    df = load_data(config.data_path)
    validate_dataframe(df, config.target_column)

    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(
        df=df,
        target_column=config.target_column,
        test_size=config.test_size,
        valid_size=config.valid_size,
        random_state=config.random_state,
    )

    categorical_cols = detect_categorical_columns(x_train)

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    train_pool = Pool(x_train, y_train, cat_features=categorical_cols)
    valid_pool = Pool(x_valid, y_valid, cat_features=categorical_cols)
    test_pool = Pool(x_test, y_test, cat_features=categorical_cols)

    model = build_model(config, scale_pos_weight=scale_pos_weight)

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    valid_prob = model.predict_proba(valid_pool)[:, 1]
    best_threshold, best_valid_f1 = find_best_threshold(y_valid, valid_prob)

    test_prob = model.predict_proba(test_pool)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_prob, best_threshold)

    feature_importance = sorted(
        zip(x_train.columns.tolist(), model.get_feature_importance(train_pool)),
        key=lambda item: item[1],
        reverse=True,
    )

    top_features = [
        {"feature": feature, "importance": round(float(importance), 6)}
        for feature, importance in feature_importance[:20]
    ]

    metadata = {
        "config": asdict(config),
        "categorical_columns": categorical_cols,
        "scale_pos_weight": round(scale_pos_weight, 6),
        "best_iteration": int(model.get_best_iteration()),
        "validation_best_f1": round(best_valid_f1, 6),
        "optimized_threshold": round(best_threshold, 6),
        "test_metrics": test_metrics,
        "top_features": top_features,
    }

    artifact = {
        "model": model,
        "threshold": best_threshold,
        "categorical_columns": categorical_cols,
        "feature_names": x_train.columns.tolist(),
        "target_column": config.target_column,
    }

    joblib.dump(artifact, config.model_output_path)
    Path(config.metadata_output_path).write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print("\nValidation best F1:", round(best_valid_f1, 4))
    print("Optimized threshold:", round(best_threshold, 4))
    print("Test accuracy:", test_metrics["accuracy"])
    print("Test F1:", test_metrics["f1"])
    print("Test ROC-AUC:", test_metrics["roc_auc"])
    print("\nConfusion Matrix:")
    print(np.array(test_metrics["confusion_matrix"]))
    print(f"\nModel saved to: {config.model_output_path}")
    print(f"Metadata saved to: {config.metadata_output_path}")


if __name__ == "__main__":
    main()
