from __future__ import annotations

import json
import random
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights


@dataclass
class Config:
    train_dir: str = "dataset/train"
    validation_dir: str = "dataset/validation"

    checkpoint_path: str = "best_efficientnet_model.pt"
    history_path: str = "training_history.json"

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

    frozen_epochs: int = 5
    fine_tune_epochs: int = 20
    early_stopping_patience: int = 5

    classifier_learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-4

    weight_decay: float = 1e-4
    dropout: float = 0.35
    label_smoothing: float = 0.1

    random_seed: int = 42


def set_seed(seed: int) -> None:
    """افزایش قابلیت بازتولید نتایج."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # این تنظیمات بازتولیدپذیری را بیشتر، ولی ممکن است آموزش را کندتر کنند.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_directories(config: Config) -> None:
    required_directories = [
        Path(config.train_dir),
        Path(config.validation_dir),
    ]

    for directory in required_directories:
        if not directory.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {directory.resolve()}"
            )


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.70, 1.0),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomPerspective(
                distortion_scale=0.15,
                p=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(
                p=0.15,
                scale=(0.02, 0.12),
            ),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return {
        "train": train_transform,
        "validation": validation_transform,
    }


def create_weighted_sampler(
    targets: list[int],
    number_of_classes: int,
) -> WeightedRandomSampler:
    """
    برای هر نمونه وزنی معکوس با فراوانی کلاس آن تولید می‌کند.
    این روش به مدیریت کلاس‌های نامتوازن کمک می‌کند.
    """

    class_counts = Counter(targets)

    class_weights = {
        class_index: len(targets) / class_counts[class_index]
        for class_index in range(number_of_classes)
    }

    sample_weights = [
        class_weights[target]
        for target in targets
    ]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(
    config: Config,
) -> tuple[DataLoader, DataLoader, datasets.ImageFolder]:
    data_transforms = build_transforms(config.image_size)

    train_dataset = datasets.ImageFolder(
        root=config.train_dir,
        transform=data_transforms["train"],
    )

    validation_dataset = datasets.ImageFolder(
        root=config.validation_dir,
        transform=data_transforms["validation"],
    )

    if train_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise ValueError(
            "Train and validation directories must contain identical classes."
        )

    if len(train_dataset.classes) < 2:
        raise ValueError("At least two classes are required.")

    sampler = create_weighted_sampler(
        targets=train_dataset.targets,
        number_of_classes=len(train_dataset.classes),
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    return train_loader, validation_loader, train_dataset


def build_model(
    number_of_classes: int,
    dropout: float,
) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(weights=weights)

    input_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(input_features, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(),
        nn.Dropout(p=dropout),
        nn.Linear(512, number_of_classes),
    )

    return model


def freeze_backbone(model: nn.Module) -> None:
    """فقط classifier آموزش داده می‌شود."""

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_blocks(
    model: nn.Module,
    number_of_blocks: int = 3,
) -> None:
    """
    ابتدا همه backbone فریز می‌شود و سپس چند بلوک انتهایی
    برای Fine-tuning آزاد می‌شوند.
    """

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    number_of_blocks = min(number_of_blocks, len(model.features))

    for block in model.features[-number_of_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mixed_precision_enabled: bool,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_predictions: list[int] = []
    all_targets: list[int] = []

    for images, targets in data_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=mixed_precision_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        # جلوگیری از انفجار گرادیان‌ها
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (predictions == targets).sum().item()

        all_predictions.extend(predictions.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_f1": f1_score(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_predictions: list[int] = []
    all_targets: list[int] = []

    for images, targets in data_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits = model(images)
        loss = criterion(logits, targets)

        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (predictions == targets).sum().item()

        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_f1": f1_score(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
    }

    return metrics, all_targets, all_predictions


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_f1: float,
    class_to_idx: dict[str, int],
    config: Config,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_macro_f1": validation_f1,
        "class_to_idx": class_to_idx,
        "config": asdict(config),
    }

    torch.save(checkpoint, path)


def run_training_stage(
    *,
    stage_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    number_of_epochs: int,
    starting_epoch: int,
    patience: int,
    class_to_idx: dict[str, int],
    config: Config,
    best_f1: float,
    best_state: dict[str, Any],
    history: list[dict[str, Any]],
) -> tuple[float, dict[str, Any], int]:
    optimizer = AdamW(
        get_trainable_parameters(model),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(number_of_epochs, 1),
        eta_min=learning_rate * 0.01,
    )

    mixed_precision_enabled = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=mixed_precision_enabled,
    )

    epochs_without_improvement = 0
    last_epoch = starting_epoch

    print(f"\n{'=' * 70}")
    print(f"Training stage: {stage_name}")
    print(f"{'=' * 70}")

    for local_epoch in range(number_of_epochs):
        epoch = starting_epoch + local_epoch + 1
        last_epoch = epoch

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixed_precision_enabled=mixed_precision_enabled,
        )

        validation_metrics, _, _ = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        current_learning_rate = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "stage": stage_name,
            "learning_rate": current_learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
        }

        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d} | "
            f"LR: {current_learning_rate:.2e} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train F1: {train_metrics['macro_f1']:.4f} | "
            f"Val loss: {validation_metrics['loss']:.4f} | "
            f"Val accuracy: {validation_metrics['accuracy']:.4f} | "
            f"Val F1: {validation_metrics['macro_f1']:.4f}"
        )

        current_f1 = validation_metrics["macro_f1"]

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0

            save_checkpoint(
                path=config.checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_f1=current_f1,
                class_to_idx=class_to_idx,
                config=config,
            )

            print(
                f"  ✓ Best checkpoint updated: "
                f"macro-F1={best_f1:.4f}"
            )

        else:
            epochs_without_improvement += 1

        scheduler.step()

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping activated after "
                f"{patience} epochs without improvement."
            )
            break

    return best_f1, best_state, last_epoch


def main() -> None:
    config = Config()

    set_seed(config.random_seed)
    validate_directories(config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, validation_loader, train_dataset = build_dataloaders(
        config
    )

    print(f"Classes: {train_dataset.classes}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_loader.dataset)}")

    model = build_model(
        number_of_classes=len(train_dataset.classes),
        dropout=config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing,
    )

    history: list[dict[str, Any]] = []
    best_state = deepcopy(model.state_dict())
    best_f1 = -1.0

    # مرحله اول: آموزش classifier
    freeze_backbone(model)

    best_f1, best_state, last_epoch = run_training_stage(
        stage_name="classifier-training",
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        device=device,
        learning_rate=config.classifier_learning_rate,
        weight_decay=config.weight_decay,
        number_of_epochs=config.frozen_epochs,
        starting_epoch=0,
        patience=config.early_stopping_patience,
        class_to_idx=train_dataset.class_to_idx,
        config=config,
        best_f1=best_f1,
        best_state=best_state,
        history=history,
    )

    # مرحله دوم: Fine-tuning بلوک‌های انتهایی EfficientNet
    unfreeze_last_blocks(
        model,
        number_of_blocks=3,
    )

    best_f1, best_state, _ = run_training_stage(
        stage_name="fine-tuning",
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        device=device,
        learning_rate=config.fine_tune_learning_rate,
        weight_decay=config.weight_decay,
        number_of_epochs=config.fine_tune_epochs,
        starting_epoch=last_epoch,
        patience=config.early_stopping_patience,
        class_to_idx=train_dataset.class_to_idx,
        config=config,
        best_f1=best_f1,
        best_state=best_state,
        history=history,
    )

    # بارگذاری بهترین وزن‌ها
    model.load_state_dict(best_state)

    final_metrics, final_targets, final_predictions = evaluate(
        model=model,
        data_loader=validation_loader,
        criterion=criterion,
        device=device,
    )

    report = classification_report(
        final_targets,
        final_predictions,
        target_names=train_dataset.classes,
        output_dict=True,
        zero_division=0,
    )

    output_history = {
        "config": asdict(config),
        "class_to_idx": train_dataset.class_to_idx,
        "best_validation_macro_f1": best_f1,
        "final_validation_metrics": final_metrics,
        "classification_report": report,
        "epochs": history,
    }

    Path(config.history_path).write_text(
        json.dumps(
            output_history,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFinal validation metrics")
    print("-" * 40)
    print(f"Loss     : {final_metrics['loss']:.4f}")
    print(f"Accuracy : {final_metrics['accuracy']:.4f}")
    print(f"Macro F1 : {final_metrics['macro_f1']:.4f}")
    print(f"Model    : {config.checkpoint_path}")
    print(f"History  : {config.history_path}")


if __name__ == "__main__":
    main()
