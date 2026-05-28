"""BGE pipeline orchestrator.

Stages:
  0. Metadata filter — predict BGE divisions (I–V) → filter corpus
  1. Bridge retrieval — map law citations → BGE decisions that cite same articles
  2. Dense retrieval — HyDE decomposition (court-focused) + Cohere + FAISS
  3. Reranking — Cohere reranker, take top-3 directly (no LLM step)

Depends on the laws pipeline: requires law_citations as input, which are the
output of run_laws_pipeline(). The bridge from law citations to BGE decisions
is an intentional design choice — BGE decisions that reference the same
statutes as the predicted law citations are likely relevant to the same query.

Returns a list of up to BGE_FINAL_TOP_K citation strings.
"""

import faiss
import numpy as np
import pandas as pd

from config import (
    BGE_DENSE_QUERY_TOP_K,
    BGE_DENSE_SUBQ_TOP_K,
    BGE_FINAL_TOP_K,
    BGE_RERANK_TOP_K,
)
from retrieval.candidate_generator import stage1_bge
from retrieval.dense_retriever import stage2_dense
from retrieval.metadata_filter import filter_bge_corpus, predict_bge_divisions
from retrieval.reranker import rerank

_BGE_RERANK_QUERY_PREFIX = (
    "Find Swiss Federal Court (BGE) decisions relevant to the following legal question. "
    "Include decisions addressing substantive law, procedural requirements, constitutional rights, "
    "or standard of review issues raised by the question.\n\n"
)


def run_bge_pipeline(
    query: str,
    law_citations: list[str],
    client,
    bge_df: pd.DataFrame,
    bge_index: faiss.IndexFlatIP,
    bge_citation_set: set[str],
    bridge: dict[str, list[str]],
    bge_divisions_json: str,
    bge_embeddings: np.ndarray | None = None,
) -> list[str]:
    """Run the full BGE retrieval pipeline for a single query.

    When bge_embeddings is provided, Stage 2 builds a filtered sub-index
    from only the predicted valid BGE citations — the global index is never
    searched. When not provided (e.g. --subset runs), falls back to
    over-fetching from the global index with post-filtering.
    """

    # Stage 0: metadata filter
    divisions = predict_bge_divisions(query, client, bge_divisions_json)
    valid_citations = filter_bge_corpus(bge_df, divisions)

    # Stage 1: bridge from law citations to BGE decisions
    s1 = stage1_bge(law_citations, bridge, bge_df)

    # Stage 2: HyDE decomposition + dense retrieval over filtered sub-index
    s2 = stage2_dense(
        query=query,
        client=client,
        index=bge_index,
        corpus_df=bge_df,
        decompose_prompt_path="retrieval/decompose_query_bge.yaml",
        subq_top_k=BGE_DENSE_SUBQ_TOP_K,
        query_top_k=BGE_DENSE_QUERY_TOP_K,
        valid_citations=valid_citations or None,
        corpus_embeddings=bge_embeddings,
    )

    # Merge: Stage 1 first, then Stage 2
    seen = set(s1)
    combined = s1 + [c for c in s2 if c not in seen]

    # Stage 3: reranking — top-3 is the final output (no LLM selection step)
    corpus_text_lookup = bge_df.set_index("citation")["text"].to_dict()
    reranked = rerank(
        query=query,
        candidates=combined,
        client=client,
        corpus_text_lookup=corpus_text_lookup,
        top_k=BGE_RERANK_TOP_K,
        query_prefix=_BGE_RERANK_QUERY_PREFIX,
    )

    return reranked[:BGE_FINAL_TOP_K]
