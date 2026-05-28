"""Docket pipeline orchestrator.

Mirrors the BGE pipeline architecture with docket-specific metadata filtering
and decomposition prompt.

Stages:
  0. Metadata filter — predict chambers + case type → filter docket corpus
  1. Bridge retrieval — map law citations → docket decisions citing same articles
  2. Dense retrieval — HyDE decomposition (docket-focused) + Cohere embed → FAISS sub-index
  3. Reranking — Cohere reranker → top-3 final

Kept separate from the BGE pipeline because docket decisions (2.22M non-leading
decisions) would add significant noise to BGE retrieval results — they are
structurally different, shorter, and far more numerous than leading BGE decisions.

Returns a list of up to DOCKET_FINAL_TOP_K citation strings.
"""

import numpy as np
import pandas as pd

from config import (
    DOCKET_DENSE_QUERY_TOP_K,
    DOCKET_DENSE_SUBQ_TOP_K,
    DOCKET_FINAL_TOP_K,
    DOCKET_RERANK_TOP_K,
)
from retrieval.candidate_generator import stage1_bge
from retrieval.dense_retriever import stage2_dense
from retrieval.metadata_filter import filter_docket_corpus, predict_docket_filters
from retrieval.reranker import rerank

_DOCKET_RERANK_QUERY_PREFIX = (
    "Find Swiss Federal Court non-leading decisions (docket citations) relevant to the following "
    "legal question. Include decisions addressing the specific procedural or substantive issue "
    "raised by the question.\n\n"
)


def run_docket_pipeline(
    query: str,
    law_citations: list[str],
    client,
    docket_df: pd.DataFrame,
    docket_index,
    docket_citation_set: set[str],
    bridge: dict[str, list[str]],
    chambers_json: str,
    procedures_json: str,
    docket_embeddings: np.ndarray | None = None,
) -> list[str]:
    """Run the full docket retrieval pipeline for a single query.

    Stage 2 builds a filtered sub-index from only the predicted valid docket
    citations. Full-corpus docket runs pass no global index, so metadata
    filtering is required before vector search.
    """

    # Stage 0: metadata filter
    filters = predict_docket_filters(query, client, chambers_json, procedures_json)
    valid_citations = filter_docket_corpus(docket_df, filters["chambers"], filters["procedures"])

    # Stage 1: bridge from law citations to docket decisions
    s1 = stage1_bge(law_citations, bridge, docket_df)

    # Stage 2: HyDE decomposition + dense retrieval over filtered sub-index
    s2 = stage2_dense(
        query=query,
        client=client,
        index=docket_index,
        corpus_df=docket_df,
        decompose_prompt_path="retrieval/decompose_query_docket.yaml",
        subq_top_k=DOCKET_DENSE_SUBQ_TOP_K,
        query_top_k=DOCKET_DENSE_QUERY_TOP_K,
        valid_citations=valid_citations or None,
        corpus_embeddings=docket_embeddings,
    )

    # Merge: Stage 1 first, then Stage 2
    seen = set(s1)
    combined = s1 + [c for c in s2 if c not in seen]

    if not combined:
        return []

    # Stage 3: reranking — top-3 is the final output
    corpus_text_lookup = docket_df.set_index("citation")["text"].to_dict()
    reranked = rerank(
        query=query,
        candidates=combined,
        client=client,
        corpus_text_lookup=corpus_text_lookup,
        top_k=DOCKET_RERANK_TOP_K,
        query_prefix=_DOCKET_RERANK_QUERY_PREFIX,
    )

    return reranked[:DOCKET_FINAL_TOP_K]
