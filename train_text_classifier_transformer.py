from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


@dataclass
class Config:
    data_path: str = "texts.csv"
    text_column: str = "text"
    label_column: str = "label"

    model_name: str = "distilbert-base-multilingual-cased"
    output_dir: str = "transformer_text_classifier"

    max_length: int = 256
    test_size: float = 0.15
    validation_size: float = 0.15
    random_seed: int = 42

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_train_epochs: int = 6
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2

    early_stopping_patience: int = 2
    label_smoothing_factor: float = 0.05
    fp16: bool = torch.cuda.is_available()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_validate_data(config: Config) -> pd.DataFrame:
    path = Path(config.data_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path)

    required_columns = {
        config.text_column,
        config.label_column,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    df = df[[config.text_column, config.label_column]].copy()
    df[config.text_column] = df[config.text_column].astype(str).str.strip()
    df[config.label_column] = df[config.label_column].astype(str).str.strip()

    df = df[
        (df[config.text_column] != "")
        & (df[config.label_column] != "")
    ].reset_index(drop=True)

    if len(df) < 20:
        raise ValueError("Dataset is too small. At least 20 rows are recommended.")

    if df[config.label_column].nunique() < 2:
        raise ValueError("At least two distinct labels are required.")

    return df


def stratified_split(
    df: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        stratify=df[config.label_column],
        random_state=config.random_seed,
    )

    validation_ratio_adjusted = config.validation_size / (1 - config.test_size)

    train_df, validation_df = train_test_split(
        train_df,
        test_size=validation_ratio_adjusted,
        stratify=train_df[config.label_column],
        random_state=config.random_seed,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        encoded["labels"] = self.labels[index]
        return encoded


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    class_counts = np.bincount(labels)
    total = class_counts.sum()

    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_function = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_function = nn.CrossEntropyLoss()

        loss = loss_function(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
    }


def save_report(
    output_dir: str,
    config: Config,
    label_encoder: LabelEncoder,
    predictions: np.ndarray,
    true_labels: np.ndarray,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    idx_to_label = {
        int(index): str(label)
        for index, label in enumerate(label_encoder.classes_)
    }

    report = {
        "config": asdict(config),
        "labels": idx_to_label,
        "classification_report": classification_report(
            true_labels,
            predictions,
            target_names=label_encoder.classes_.tolist(),
            output_dict=True,
            zero_division=0,
        ),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_f1": float(f1_score(true_labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(true_labels, predictions, average="weighted")),
    }

    (output_path / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    config = Config()

    set_seed(config.random_seed)

    df = load_and_validate_data(config)

    train_df, validation_df, test_df = stratified_split(df, config)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[config.label_column])
    y_validation = label_encoder.transform(validation_df[config.label_column])
    y_test = label_encoder.transform(test_df[config.label_column])

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_dataset = TextClassificationDataset(
        texts=train_df[config.text_column].tolist(),
        labels=y_train.tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    validation_dataset = TextClassificationDataset(
        texts=validation_df[config.text_column].tolist(),
        labels=y_validation.tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    test_dataset = TextClassificationDataset(
        texts=test_df[config.text_column].tolist(),
        labels=y_test.tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_encoder.classes_),
        id2label={
            i: label for i, label in enumerate(label_encoder.classes_)
        },
        label2id={
            label: i for i, label in enumerate(label_encoder.classes_)
        },
    )

    class_weights = compute_class_weights(y_train)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_dir=str(Path(config.output_dir) / "logs"),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=config.fp16,
        report_to="none",
        save_total_limit=2,
        label_smoothing_factor=config.label_smoothing_factor,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ],
    )

    trainer.train()

    evaluation = trainer.evaluate(test_dataset)
    print("Test metrics:")
    for key, value in evaluation.items():
        print(f"{key}: {value}")

    predictions_output = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions_output.predictions, axis=1)

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    metadata = {
        "config": asdict(config),
        "classes": label_encoder.classes_.tolist(),
    }

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    (Path(config.output_dir) / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_report(
        output_dir=config.output_dir,
        config=config,
        label_encoder=label_encoder,
        predictions=predicted_labels,
        true_labels=y_test,
    )

    print(f"Model artifacts saved in: {config.output_dir}")


if __name__ == "__main__":
    main()
