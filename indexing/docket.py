"""Offline indexing for the docket (non-leading Federal Court decisions) corpus.

Filters court_considerations.csv to non-BGE rows, parses docket metadata
(chamber, casetype), and embeds texts via Cohere multilingual embeddings.

Run once before using the docket pipeline.

Usage:
    python -m indexing.docket
    python -m indexing.docket --subset 1000  # for local testing
"""

import argparse
import pickle

import numpy as np
import pandas as pd

from config import INDICES_DIR, RAW_DATA_DIR
from providers import bedrock
from retrieval.candidate_generator import build_law_to_court_bridge
from retrieval.dense_retriever import embed_texts


def load_docket_df(subset: int | None = None) -> pd.DataFrame:
    path = RAW_DATA_DIR / "court_considerations.csv"
    df = pd.read_csv(path, usecols=["citation", "text"])
    df["citation"] = df["citation"].astype(str)
    df = df[~df["citation"].str.startswith("BGE")].reset_index(drop=True)

    cit = df["citation"]
    df["prefix"] = (
        cit.str.extract(r"^([A-Z0-9]+)_", expand=False)
        .fillna(cit.str.extract(r"^([A-Z0-9]+)\.\d+/", expand=False))
        .fillna(cit.str.extract(r"^([A-Z]+)\s+\d+/", expand=False))
    )
    df["chamber"] = df["prefix"].str.extract(r"^(\d+)", expand=False)
    df["casetype"] = df["prefix"].str.extract(r"([A-Za-z]+)", expand=False)

    # Drop old-format citations without a numeric chamber (no gold citations lost)
    df = df[df["chamber"].notna()].reset_index(drop=True)

    # Base decision identifier for decision-level expansion (mirrors bge_base in BGE)
    # e.g. "5A_800/2019 E. 2." → base "5A_800/2019"
    df["bge_base"] = df["citation"].str.extract(r"^([^\s]+(?:/\d{4})?)")

    if subset:
        df = df.head(subset)

    df["embedding_text"] = (
        df["citation"].fillna("").str.strip()
        + "\n\n"
        + df["text"].fillna("").str.strip()
    )
    return df


def build_docket_embeddings(df: pd.DataFrame, client) -> np.ndarray:
    return embed_texts(df["embedding_text"].tolist(), client)


def save_docket_embeddings(embeddings: np.ndarray) -> None:
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDICES_DIR / "docket_embeddings.npy", embeddings)
    print(f"Saved docket embeddings: {len(embeddings):,} vectors → {INDICES_DIR}")


def save_docket_bridge(bridge: dict) -> None:
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDICES_DIR / "docket_bridge.pkl", "wb") as f:
        pickle.dump(bridge, f)
    print(f"Saved docket bridge: {len(bridge):,} law citation keys → {INDICES_DIR}")


def load_docket_embeddings() -> np.ndarray | None:
    emb_path = INDICES_DIR / "docket_embeddings.npy"
    if emb_path.exists():
        return np.load(str(emb_path), mmap_mode="r")
    return None


def load_docket_bridge() -> dict | None:
    bridge_path = INDICES_DIR / "docket_bridge.pkl"
    if bridge_path.exists():
        with open(bridge_path, "rb") as f:
            return pickle.load(f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    client = bedrock.get_client()
    print("Loading docket corpus...")
    df = load_docket_df(subset=args.subset)
    print(f"Docket rows: {len(df):,}")

    print("Embedding...")
    embeddings = build_docket_embeddings(df, client)
    save_docket_embeddings(embeddings)

    laws_path = RAW_DATA_DIR / "laws_de.csv"
    laws_citation_set = set(pd.read_csv(laws_path, usecols=["citation"])["citation"].astype(str))
    print("Building law-to-docket bridge...")
    bridge = build_law_to_court_bridge(df, laws_citation_set)
    save_docket_bridge(bridge)
