"""Stage 2: HyDE query decomposition + dense retrieval via Cohere embeddings + FAISS.

Used by the laws, BGE, and docket pipelines. The pipeline:
  1. Decompose the English query into 5 German HyDE sub-questions + hypothetical answers
  2. Embed each sub-question+answer pair and retrieve top-k from FAISS
  3. Embed the original query and retrieve top-k
  4. Merge, deduplicate, and apply metadata filter

Cohere multilingual embeddings are language-agnostic at the vector level, so
German HyDE text can match French/Italian court decisions without needing
separate per-language embeddings.
"""

import gc
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import (
    COHERE_EMBED_MODEL,
    EMBED_BATCH_SIZE,
    EMBED_MAX_CHARS,
    EMBED_NUM_WORKERS,
    NOVA_PRO_MODEL,
    load_prompt,
)
from providers import bedrock


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], client, input_type: str = "search_document") -> np.ndarray:
    """Embed a list of texts in batches using Cohere multilingual embeddings."""
    batches = [(i, texts[i:i + EMBED_BATCH_SIZE]) for i in range(0, len(texts), EMBED_BATCH_SIZE)]
    result: np.ndarray | None = None

    def _embed_batch(args):
        start, batch = args
        body = {
            "texts": [t[:EMBED_MAX_CHARS] for t in batch],
            "input_type": input_type,
            "truncate": "END",
        }
        result = bedrock.invoke(client, COHERE_EMBED_MODEL, body)
        batch_embeddings = result.get("embeddings", [])
        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                f"Batch starting at {start}: expected {len(batch)} embeddings, got {len(batch_embeddings)}"
            )
        return start, np.asarray(batch_embeddings, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=EMBED_NUM_WORKERS) as executor:
        futures = [executor.submit(_embed_batch, b) for b in batches]
        for future in tqdm(as_completed(futures), total=len(batches), desc="Embedding"):
            start, embeddings = future.result()
            if result is None:
                result = np.empty((len(texts), embeddings.shape[1]), dtype=np.float32)
            result[start:start + len(embeddings)] = embeddings

    if result is None:
        return np.empty((0, 0), dtype=np.float32)
    if len(result) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(result)}")
    return result


def embed_query(text: str, client) -> np.ndarray:
    """Embed a single query string, L2-normalized for cosine similarity."""
    body = {
        "texts": [text[:EMBED_MAX_CHARS]],
        "input_type": "search_query",
    }
    result = bedrock.invoke(client, COHERE_EMBED_MODEL, body)
    vec = np.array(result.get("embeddings", [[]]), dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an L2-normalized inner-product FAISS index."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    normed = embeddings.copy()
    faiss.normalize_L2(normed)
    index.add(normed)
    return index


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------

def decompose_query(query: str, prompt_path: str, client) -> list[dict]:
    """Decompose query into 5 German HyDE sub-question/answer pairs.

    prompt_path selects between laws and BGE decomposition prompts, which
    differ only in their domain framing (statute text vs court reasoning).
    """
    prompt = load_prompt(prompt_path, query=query)
    raw = bedrock.converse(
        client,
        model_id=NOVA_PRO_MODEL,
        system_prompt=prompt["system"],
        user_message=prompt["user"],
        max_tokens=2048,
    )
    try:
        return json.loads(raw)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# FAISS retrieval
# ---------------------------------------------------------------------------

def retrieve_from_index(
    query_text: str,
    client,
    index: faiss.IndexFlatIP | None,
    corpus_df: pd.DataFrame,
    top_k: int,
    valid_citations: set[str] | None = None,
    corpus_embeddings: np.ndarray | None = None,
) -> list[str]:
    """Embed a query and retrieve top-k citations from a FAISS index.

    When valid_citations and corpus_embeddings are both provided, a filtered
    sub-index is built from only the valid rows — the global index is never
    searched. This is true pre-filtering: the retrieval space is reduced before
    any search happens.

    When corpus_embeddings is not available (e.g. --subset runs), falls back
    to over-fetching from the global index and post-filtering. If neither a
    filtered embedding set nor a global index is available, retrieval fails
    loudly because there is no searchable vector source.
    """
    vec = embed_query(query_text, client)

    if valid_citations and corpus_embeddings is not None:
        valid_mask = corpus_df["citation"].isin(valid_citations)
        valid_row_positions = np.flatnonzero(valid_mask.to_numpy())
        if len(valid_row_positions) == 0:
            return []
        filtered_embs = corpus_embeddings[valid_row_positions].copy()
        faiss.normalize_L2(filtered_embs)
        sub_index = faiss.IndexFlatIP(filtered_embs.shape[1])
        sub_index.add(filtered_embs)
        k = min(top_k, len(valid_row_positions))
        _, sub_indices = sub_index.search(vec, k)
        results = [corpus_df.iloc[valid_row_positions[i]]["citation"] for i in sub_indices[0] if i >= 0]
        del filtered_embs, sub_index
        gc.collect()
        return results

    if index is None:
        raise ValueError("Dense retrieval requires corpus_embeddings for pre-filtered search or a global index fallback.")

    # Fallback: over-fetch from global index and post-filter
    search_k = top_k * 3 if valid_citations else top_k
    _, indices = index.search(vec, search_k)
    results = []
    for idx in indices[0]:
        if idx < 0:
            continue
        citation = corpus_df.iloc[idx]["citation"]
        if valid_citations is None or citation in valid_citations:
            results.append(citation)
        if len(results) == top_k:
            break
    return results


def stage2_dense(
    query: str,
    client,
    index: faiss.IndexFlatIP | None,
    corpus_df: pd.DataFrame,
    decompose_prompt_path: str,
    subq_top_k: int,
    query_top_k: int,
    valid_citations: set[str] | None = None,
    corpus_embeddings: np.ndarray | None = None,
) -> list[str]:
    """Run HyDE decomposition + FAISS retrieval and return deduplicated candidates.

    Retrieves from 5 sub-questions (subq_top_k each) + the original query
    (query_top_k), then deduplicates preserving order.

    When corpus_embeddings is provided, each retrieval call builds a filtered
    sub-index from valid_citations so the global index is never searched.
    """
    pairs = decompose_query(query, decompose_prompt_path, client)
    all_candidates: list[str] = []

    for pair in pairs:
        retrieval_text = pair.get("question", "") + "\n\n" + pair.get("answer", "")
        all_candidates.extend(
            retrieve_from_index(
                retrieval_text, client, index, corpus_df, subq_top_k,
                valid_citations, corpus_embeddings,
            )
        )

    all_candidates.extend(
        retrieve_from_index(
            query, client, index, corpus_df, query_top_k,
            valid_citations, corpus_embeddings,
        )
    )

    seen: set[str] = set()
    return [c for c in all_candidates if not (c in seen or seen.add(c))]
