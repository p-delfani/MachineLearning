from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from optuna.samplers import TPESampler


@dataclass
class Config:
    data_path: str = "train_data.csv"
    target_column: str = "target"
    artifact_path: str = "lightgbm_optuna_artifact.joblib"
    report_path: str = "lightgbm_optuna_report.json"
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5
    n_trials: int = 30


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


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    params: dict,
    random_state: int,
) -> Pipeline:
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=1000,
        random_state=random_state,
        n_jobs=-1,
        **params,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )
    return pipeline


def evaluate_model(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "f1": round(f1_score(y_true, y_pred), 6),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 6),
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

    cv = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        pipeline = build_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            params=params,
            random_state=config.random_state,
        )

        oof_prob = cross_val_predict(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        return roc_auc_score(y_train, oof_prob)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.random_state),
    )
    study.optimize(objective, n_trials=config.n_trials)

    best_params = study.best_params

    final_pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        params=best_params,
        random_state=config.random_state,
    )

    final_pipeline.fit(x_train, y_train)

    test_prob = final_pipeline.predict_proba(x_test)[:, 1]
    metrics = evaluate_model(y_test, test_prob, threshold=0.5)

    preprocessor = final_pipeline.named_steps["preprocessor"]
    model = final_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out().tolist()
    importances = model.feature_importances_.tolist()

    top_features = sorted(
        [
            {"feature": feature, "importance": int(importance)}
            for feature, importance in zip(feature_names, importances)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )[:25]

    artifact = {
        "pipeline": final_pipeline,
        "target_column": config.target_column,
        "threshold": 0.5,
        "feature_names": x.columns.tolist(),
    }

    report = {
        "config": asdict(config),
        "best_params": best_params,
        "best_cv_score": round(study.best_value, 6),
        "test_metrics": metrics,
        "top_features": top_features,
    }

    joblib.dump(artifact, config.artifact_path)
    Path(config.report_path).write_text(
        json.dumps(report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print("Best CV ROC-AUC:", round(study.best_value, 4))
    print("Best Params:", best_params)
    print("Test Accuracy:", metrics["accuracy"])
    print("Test F1:", metrics["f1"])
    print("Test ROC-AUC:", metrics["roc_auc"])
    print(f"Artifact saved to: {config.artifact_path}")
    print(f"Report saved to: {config.report_path}")


if __name__ == "__main__":
    main()
