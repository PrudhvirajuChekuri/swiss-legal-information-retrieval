"""Stage 3: Cohere reranker — shared by laws and BGE pipelines."""

import json

from config import COHERE_RERANK_MODEL, EMBED_MAX_CHARS
from providers import bedrock

COHERE_RERANK_MAX_DOCUMENTS = 1000


def rerank(
    query: str,
    candidates: list[str],
    client,
    corpus_text_lookup: dict[str, str],
    top_k: int,
    query_prefix: str = "",
) -> list[str]:
    """Rerank candidates using Cohere reranker and return top_k citations.

    corpus_text_lookup maps citation → text for building the document list.
    query_prefix lets callers prepend retrieval instructions to the query
    (used by the BGE pipeline to add domain framing).
    """
    if not candidates:
        return []

    candidates = candidates[:COHERE_RERANK_MAX_DOCUMENTS]
    rerank_query = (query_prefix + query) if query_prefix else query

    documents = [
        corpus_text_lookup.get(c, c)[:EMBED_MAX_CHARS]
        for c in candidates
    ]

    body = {
        "api_version": 2,
        "query": rerank_query,
        "documents": documents,
        "top_n": min(top_k, len(candidates)),
    }

    result = bedrock.invoke(client, COHERE_RERANK_MODEL, body)
    results = result.get("results", [])
    return [candidates[r["index"]] for r in results]
