"""Offline indexing for the BGE (leading Federal Court decisions) corpus.

Filters court_considerations.csv to BGE rows, embeds them, builds a FAISS
index, and saves the law-to-BGE bridge map (requires laws corpus citation set).

Run once before using the BGE pipeline.

Usage:
    python -m indexing.bge
    python -m indexing.bge --subset 1000  # for local testing
"""

import argparse
import pickle

import faiss
import numpy as np
import pandas as pd

from config import INDICES_DIR, RAW_DATA_DIR
from providers import bedrock
from retrieval.candidate_generator import build_law_to_court_bridge
from retrieval.dense_retriever import build_faiss_index, embed_texts


def load_bge_df(subset: int | None = None) -> pd.DataFrame:
    path = RAW_DATA_DIR / "court_considerations.csv"
    df = pd.read_csv(path, usecols=["citation", "text"])
    df = df[df["citation"].astype(str).str.startswith("BGE")].reset_index(drop=True)
    if subset:
        df = df.head(subset)
    df["division"] = df["citation"].str.extract(r"BGE\s+\d+\s+(\S+)\s+\d+")
    df["bge_base"] = df["citation"].str.extract(r"(BGE\s+\d+\s+\S+\s+\d+)")
    df["embedding_text"] = (
        df["citation"].fillna("").str.strip()
        + "\n\n"
        + df["text"].fillna("").str.strip()
    )
    return df


def build_bge_index(df: pd.DataFrame, client) -> tuple[np.ndarray, faiss.IndexFlatIP]:
    embeddings = embed_texts(df["embedding_text"].tolist(), client)
    index = build_faiss_index(embeddings)
    return embeddings, index


def save_bge_index(
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    bridge: dict,
) -> None:
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDICES_DIR / "bge_embeddings.npy", embeddings)
    faiss.write_index(index, str(INDICES_DIR / "bge_faiss.index"))
    with open(INDICES_DIR / "bge_bridge.pkl", "wb") as f:
        pickle.dump(bridge, f)
    print(f"Saved BGE index: {index.ntotal} vectors → {INDICES_DIR}")


def load_bge_index() -> faiss.IndexFlatIP | None:
    index_path = INDICES_DIR / "bge_faiss.index"
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return None


def load_bge_bridge() -> dict | None:
    bridge_path = INDICES_DIR / "bge_bridge.pkl"
    if bridge_path.exists():
        with open(bridge_path, "rb") as f:
            return pickle.load(f)
    return None


def load_bge_embeddings() -> np.ndarray | None:
    emb_path = INDICES_DIR / "bge_embeddings.npy"
    if emb_path.exists():
        return np.load(str(emb_path))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    client = bedrock.get_client()

    print("Loading BGE corpus...")
    bge_df = load_bge_df(subset=args.subset)
    print(f"BGE rows: {len(bge_df):,}")

    laws_path = RAW_DATA_DIR / "laws_de.csv"
    laws_citation_set = set(pd.read_csv(laws_path, usecols=["citation"])["citation"].astype(str))

    print("Building law-to-BGE bridge...")
    bridge = build_law_to_court_bridge(bge_df, laws_citation_set)
    print(f"Bridge entries: {len(bridge):,}")

    print("Embedding BGE corpus...")
    embeddings, index = build_bge_index(bge_df, client)
    save_bge_index(embeddings, index, bridge)
