
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class Config:
    data_path: str = "documents.csv"

    id_column: str = "id"
    title_column: str = "title"
    text_column: str = "text"

    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    output_dir: str = "semantic_search_index"
    index_file: str = "faiss.index"
    documents_file: str = "documents.parquet"
    metadata_file: str = "metadata.json"

    batch_size: int = 64
    normalize_embeddings: bool = True

    random_seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_and_validate_documents(config: Config) -> pd.DataFrame:
    path = Path(config.data_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path)

    required_columns = {
        config.id_column,
        config.title_column,
        config.text_column,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df[
        [
            config.id_column,
            config.title_column,
            config.text_column,
        ]
    ].copy()

    df[config.title_column] = df[config.title_column].astype(str).str.strip()
    df[config.text_column] = df[config.text_column].astype(str).str.strip()

    df = df[
        (df[config.title_column] != "")
        & (df[config.text_column] != "")
    ].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid documents found after cleaning.")

    df["combined_text"] = (
        df[config.title_column]
        + "\n"
        + df[config.text_column]
    )

    return df


def create_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    embedding_dimension = embeddings.shape[1]

    # چون embeddingها normalize شده‌اند، Inner Product معادل cosine similarity است.
    index = faiss.IndexFlatIP(embedding_dimension)

    index.add(embeddings)

    return index


def save_artifacts(
    config: Config,
    index: faiss.Index,
    documents: pd.DataFrame,
    embedding_dimension: int,
) -> None:
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(
        index,
        str(output_path / config.index_file),
    )

    documents.to_parquet(
        output_path / config.documents_file,
        index=False,
    )

    metadata: dict[str, Any] = {
        "config": asdict(config),
        "number_of_documents": int(len(documents)),
        "embedding_dimension": int(embedding_dimension),
        "similarity_metric": "cosine_similarity",
        "faiss_index_type": "IndexFlatIP",
    }

    (output_path / config.metadata_file).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    config = Config()
    set_seed(config.random_seed)

    print("Loading documents...")
    documents = load_and_validate_documents(config)

    print(f"Documents: {len(documents):,}")

    print("Loading embedding model...")
    model = SentenceTransformer(config.model_name)

    print("Creating embeddings...")
    embeddings = create_embeddings(
        model=model,
        texts=documents["combined_text"].tolist(),
        batch_size=config.batch_size,
        normalize_embeddings=config.normalize_embeddings,
    )

    print(f"Embedding shape: {embeddings.shape}")

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    save_artifacts(
        config=config,
        index=index,
        documents=documents,
        embedding_dimension=embeddings.shape[1],
    )

    print("\nSemantic search index created successfully.")
    print(f"Output directory: {config.output_dir}")
    print(f"FAISS vectors: {index.ntotal:,}")


if __name__ == "__main__":
    main()
