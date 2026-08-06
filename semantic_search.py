
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class SearchConfig:
    index_dir: str = "semantic_search_index"

    index_file: str = "faiss.index"
    documents_file: str = "documents.parquet"
    metadata_file: str = "metadata.json"

    top_k: int = 5
    min_score: float = 0.20


def load_search_artifacts(
    config: SearchConfig,
) -> tuple[faiss.Index, pd.DataFrame, dict[str, Any], SentenceTransformer]:
    index_path = Path(config.index_dir)

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index directory not found: {index_path.resolve()}"
        )

    metadata = json.loads(
        (index_path / config.metadata_file).read_text(encoding="utf-8")
    )

    model_name = metadata["config"]["model_name"]

    index = faiss.read_index(str(index_path / config.index_file))

    documents = pd.read_parquet(index_path / config.documents_file)

    model = SentenceTransformer(model_name)

    return index, documents, metadata, model


def encode_query(
    model: SentenceTransformer,
    query: str,
) -> np.ndarray:
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding.astype("float32")


def semantic_search(
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    config = SearchConfig()

    if top_k is not None:
        config.top_k = top_k

    if min_score is not None:
        config.min_score = min_score

    index, documents, _, model = load_search_artifacts(config)

    query_embedding = encode_query(model, query)

    scores, indices = index.search(
        query_embedding,
        config.top_k,
    )

    results: list[dict[str, Any]] = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position == -1:
            continue

        if float(score) < config.min_score:
            continue

        row = documents.iloc[int(index_position)]

        results.append(
            {
                "score": float(score),
                "id": row["id"],
                "title": row["title"],
                "text": row["text"],
            }
        )

    return results


def print_results(query: str, results: list[dict[str, Any]]) -> None:
    print("=" * 80)
    print(f"Query: {query}")
    print("=" * 80)

    if not results:
        print("No relevant documents found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("-" * 80)
        print(f"Score : {result['score']:.4f}")
        print(f"ID    : {result['id']}")
        print(f"Title : {result['title']}")
        print(f"Text  : {result['text']}")


def main() -> None:
    query = input("Search query: ").strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    results = semantic_search(
        query=query,
        top_k=5,
        min_score=0.20,
    )

    print_results(query, results)


if __name__ == "__main__":
    main()
