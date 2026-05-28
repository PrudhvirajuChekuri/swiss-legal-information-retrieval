"""Laws pipeline orchestrator.

Stages:
  0. Metadata filter — predict law codes → filter corpus to valid_citations
  1. Direct candidate generation — extract explicit citations, expand by Abs. + title
  2. Dense retrieval — HyDE decomposition + Cohere embeddings + FAISS
  3. Reranking — Cohere reranker
  4. LLM final selection — Claude selects directly relevant citations

Returns a list of citation strings.
"""

import faiss
import numpy as np
import pandas as pd

from config import (
    LAWS_DENSE_QUERY_TOP_K,
    LAWS_DENSE_SUBQ_TOP_K,
    LAWS_RERANK_TOP_K,
)
from retrieval.candidate_generator import stage1_laws
from retrieval.dense_retriever import stage2_dense
from retrieval.llm_selector import final_llm_selection
from retrieval.metadata_filter import filter_laws_corpus, predict_law_codes
from retrieval.reranker import rerank


def run_laws_pipeline(
    query: str,
    client,
    laws_df: pd.DataFrame,
    laws_index: faiss.IndexFlatIP,
    laws_citation_set: set[str],
    law_codes_json: str,
    laws_embeddings: np.ndarray | None = None,
) -> list[str]:
    """Run the full laws retrieval pipeline for a single query.

    When laws_embeddings is provided, Stage 2 builds a filtered sub-index
    from only the predicted valid law citations — the global index is never
    searched. When not provided (e.g. --subset runs), falls back to
    over-fetching from the global index with post-filtering.
    """

    # Stage 0: metadata filter
    law_codes = predict_law_codes(query, client, law_codes_json)
    valid_citations = filter_laws_corpus(laws_df, law_codes)

    # Stage 1: direct candidate generation
    s1 = stage1_laws(query, client, laws_df, laws_citation_set)

    # Stage 2: HyDE decomposition + dense retrieval over filtered sub-index
    s2 = stage2_dense(
        query=query,
        client=client,
        index=laws_index,
        corpus_df=laws_df,
        decompose_prompt_path="retrieval/decompose_query.yaml",
        subq_top_k=LAWS_DENSE_SUBQ_TOP_K,
        query_top_k=LAWS_DENSE_QUERY_TOP_K,
        valid_citations=valid_citations or None,
        corpus_embeddings=laws_embeddings,
    )

    # Merge: Stage 1 first (higher confidence), then Stage 2
    seen = set(s1)
    combined = s1 + [c for c in s2 if c not in seen]

    # Stage 3: reranking
    corpus_text_lookup = laws_df.set_index("citation")["embedding_text"].to_dict()
    reranked = rerank(
        query=query,
        candidates=combined,
        client=client,
        corpus_text_lookup=corpus_text_lookup,
        top_k=LAWS_RERANK_TOP_K,
    )

    # Stage 4: LLM final selection
    corpus_full_text = laws_df.set_index("citation")["text"].to_dict()
    selected = final_llm_selection(
        query=query,
        candidates=reranked,
        client=client,
        corpus_text_lookup=corpus_full_text,
        citation_set=laws_citation_set,
    )

    return selected
