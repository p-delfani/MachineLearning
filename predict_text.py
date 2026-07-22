from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = "transformer_text_classifier"
TEXTS = [
    "این محصول کیفیت خیلی خوبی داشت",
    "پشتیبانی اصلاً پاسخگو نبود و تجربه بدی داشتم",
]


def load_artifacts(model_dir: str):
    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path.resolve()}")

    metadata = json.loads(
        (model_path / "metadata.json").read_text(encoding="utf-8")
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    classes = metadata["classes"]

    return tokenizer, model, classes, device


@torch.inference_mode()
def predict(texts: list[str]) -> None:
    tokenizer, model, classes, device = load_artifacts(MODEL_DIR)

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    ).to(device)

    logits = model(**encoded).logits
    probabilities = torch.softmax(logits, dim=1)

    predicted_indices = probabilities.argmax(dim=1)

    for text, probs, index in zip(texts, probabilities, predicted_indices):
        print("-" * 60)
        print(f"Text: {text}")
        print(f"Predicted label: {classes[index.item()]}")
        print("Top probabilities:")

        top_values, top_indices = torch.topk(
            probs,
            k=min(3, len(classes)),
        )

        for score, class_index in zip(top_values, top_indices):
            print(f"  {classes[class_index.item()]}: {score.item() * 100:.2f}%")


if __name__ == "__main__":
    predict(TEXTS)
