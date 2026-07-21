
from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


# =========================================================
# Configuration
# =========================================================

@dataclass
class Config:
    data_path: str = "timeseries.csv"

    timestamp_column: str = "timestamp"
    feature_columns: tuple[str, ...] = (
        "value",
        "temperature",
    )

    # اختیاری؛ اگر این ستون وجود نداشته باشد، فقط تشخیص انجام می‌شود.
    label_column: str = "is_anomaly"

    window_size: int = 30
    window_stride: int = 1

    train_ratio: float = 0.70
    validation_ratio: float = 0.15

    batch_size: int = 128
    num_workers: int = 0

    hidden_size: int = 64
    latent_size: int = 24
    number_of_layers: int = 2
    dropout: float = 0.20

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    maximum_epochs: int = 60
    early_stopping_patience: int = 8

    # آستانه بر اساس صدک خطای بازسازی Validation
    threshold_percentile: float = 99.0

    model_path: str = "lstm_autoencoder.pt"
    scaler_path: str = "timeseries_scaler.joblib"
    metadata_path: str = "anomaly_metadata.json"
    predictions_path: str = "anomaly_predictions.csv"

    random_seed: int = 42


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


# =========================================================
# Data validation and preprocessing
# =========================================================

def load_and_validate_data(config: Config) -> pd.DataFrame:
    data_path = Path(config.data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path.resolve()}"
        )

    dataframe = pd.read_csv(data_path)

    required_columns = {
        config.timestamp_column,
        *config.feature_columns,
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    dataframe[config.timestamp_column] = pd.to_datetime(
        dataframe[config.timestamp_column],
        errors="coerce",
    )

    if dataframe[config.timestamp_column].isna().any():
        invalid_count = int(
            dataframe[config.timestamp_column].isna().sum()
        )

        raise ValueError(
            f"{invalid_count} invalid timestamps were detected."
        )

    dataframe = (
        dataframe
        .sort_values(config.timestamp_column)
        .drop_duplicates(
            subset=[config.timestamp_column],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # تبدیل ویژگی‌ها به مقدار عددی
    for column in config.feature_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    # Interpolation تنها با اطلاعات نقاط اطراف انجام می‌شود.
    dataframe[list(config.feature_columns)] = (
        dataframe[list(config.feature_columns)]
        .interpolate(method="linear", limit_direction="both")
    )

    if dataframe[list(config.feature_columns)].isna().any().any():
        raise ValueError(
            "Missing values remain after interpolation."
        )

    if len(dataframe) < config.window_size * 4:
        raise ValueError(
            "Dataset is too small for the configured window size."
        )

    if config.label_column in dataframe.columns:
        dataframe[config.label_column] = (
            pd.to_numeric(
                dataframe[config.label_column],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

        invalid_labels = set(
            dataframe[config.label_column].unique()
        ).difference({0, 1})

        if invalid_labels:
            raise ValueError(
                "The anomaly label must contain only 0 and 1."
            )

    return dataframe


def chronological_split(
    dataframe: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < config.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < config.validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if config.train_ratio + config.validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be smaller than 1."
        )

    total_rows = len(dataframe)

    train_end = int(total_rows * config.train_ratio)
    validation_end = int(
        total_rows
        * (config.train_ratio + config.validation_ratio)
    )

    train_df = dataframe.iloc[:train_end].copy()
    validation_df = dataframe.iloc[
        train_end:validation_end
    ].copy()
    test_df = dataframe.iloc[validation_end:].copy()

    minimum_size = config.window_size

    for name, split in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        if len(split) < minimum_size:
            raise ValueError(
                f"The {name} split contains fewer rows than window_size."
            )

    return train_df, validation_df, test_df


def fit_and_apply_scaler(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """
    Scaler فقط روی Train آموزش می‌بیند تا از Data Leakage
    جلوگیری شود.
    """

    scaler = RobustScaler()

    train_values = scaler.fit_transform(
        train_df[list(feature_columns)]
    )

    validation_values = scaler.transform(
        validation_df[list(feature_columns)]
    )

    test_values = scaler.transform(
        test_df[list(feature_columns)]
    )

    return (
        train_values.astype(np.float32),
        validation_values.astype(np.float32),
        test_values.astype(np.float32),
        scaler,
    )


# =========================================================
# Window creation
# =========================================================

def create_windows(
    values: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    خروجی دوم، اندیس نقطه انتهایی هر پنجره است.
    """

    windows: list[np.ndarray] = []
    endpoint_indices: list[int] = []

    for start in range(
        0,
        len(values) - window_size + 1,
        stride,
    ):
        end = start + window_size

        windows.append(values[start:end])
        endpoint_indices.append(end - 1)

    if not windows:
        raise ValueError(
            "No windows were created. Reduce window_size."
        )

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(endpoint_indices, dtype=np.int64),
    )


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, windows: np.ndarray) -> None:
        self.windows = torch.from_numpy(windows).float()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        # در Autoencoder ورودی، هدف بازسازی خودش است.
        return self.windows[index]


def create_data_loader(
    windows: np.ndarray,
    batch_size: int,
    shuffle: bool,
    number_of_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = TimeSeriesWindowDataset(windows)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=number_of_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=number_of_workers > 0,
        drop_last=False,
    )


# =========================================================
# LSTM Autoencoder
# =========================================================

class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        number_of_features: int,
        hidden_size: int,
        latent_size: int,
        number_of_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        effective_dropout = (
            dropout if number_of_layers > 1 else 0.0
        )

        self.encoder = nn.LSTM(
            input_size=number_of_features,
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.to_latent = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.LayerNorm(latent_size),
            nn.Tanh(),
        )

        self.from_latent = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            number_of_features,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.size(1)

        _, (hidden_state, _) = self.encoder(inputs)

        # hidden state مربوط به آخرین لایه Encoder
        encoded = hidden_state[-1]

        latent = self.to_latent(encoded)
        decoder_seed = self.from_latent(latent)

        repeated_context = decoder_seed.unsqueeze(1).repeat(
            1,
            sequence_length,
            1,
        )

        decoded_sequence, _ = self.decoder(repeated_context)

        reconstructed = self.output_layer(decoded_sequence)

        return reconstructed


# =========================================================
# Training and evaluation
# =========================================================

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mixed_precision_enabled: bool,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for windows in data_loader:
        windows = windows.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=mixed_precision_enabled,
        ):
            reconstructed = model(windows)
            loss = loss_function(reconstructed, windows)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        batch_size = windows.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.inference_mode()
def calculate_reconstruction_errors(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> 
