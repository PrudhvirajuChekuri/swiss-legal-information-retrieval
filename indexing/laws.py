"""Offline indexing for the laws corpus.

Embeds laws_de.csv and saves FAISS index to indices/.
Run once before using the laws pipeline.

Usage:
    python -m indexing.laws
    python -m indexing.laws --subset 1000  # for local testing
"""

import argparse
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from config import INDICES_DIR, MAPPINGS_DIR, RAW_DATA_DIR
from providers import bedrock
from retrieval.dense_retriever import build_faiss_index, embed_texts


def load_laws_df(subset: int | None = None) -> pd.DataFrame:
    path = RAW_DATA_DIR / "laws_de.csv"
    df = pd.read_csv(path)
    if subset:
        df = df.head(subset)
    df["law_code"] = df["citation"].apply(_extract_law_code)
    df["embedding_text"] = (
        df["title"].fillna("").str.strip()
        + "\n\n"
        + df["citation"].fillna("").str.strip()
        + "\n\n"
        + df["text"].fillna("").str.strip()
    )
    return df


def _extract_law_code(citation: str) -> str | None:
    m = re.search(r"Art\.\s+\d+[a-z]*(?:\s+Abs\.\s+\d+[a-z]*)*\s+(\S+)$", str(citation))
    return m.group(1) if m else None


def build_laws_index(df: pd.DataFrame, client) -> tuple[np.ndarray, faiss.IndexFlatIP]:
    embeddings = embed_texts(df["embedding_text"].tolist(), client)
    index = build_faiss_index(embeddings)
    return embeddings, index


def save_laws_index(embeddings: np.ndarray, index: faiss.IndexFlatIP) -> None:
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDICES_DIR / "laws_embeddings.npy", embeddings)
    faiss.write_index(index, str(INDICES_DIR / "laws_faiss.index"))
    print(f"Saved laws index: {index.ntotal} vectors → {INDICES_DIR}")


def load_laws_index() -> faiss.IndexFlatIP | None:
    index_path = INDICES_DIR / "laws_faiss.index"
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return None


def load_laws_embeddings() -> np.ndarray | None:
    emb_path = INDICES_DIR / "laws_embeddings.npy"
    if emb_path.exists():
        return np.load(str(emb_path))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    client = bedrock.get_client()
    print("Loading laws corpus...")
    df = load_laws_df(subset=args.subset)
    print(f"Rows: {len(df):,}")

    print("Embedding...")
    embeddings, index = build_laws_index(df, client)
    save_laws_index(embeddings, index)
