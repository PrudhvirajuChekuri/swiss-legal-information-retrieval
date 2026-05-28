"""Stage 4: Claude final relevance filter — used by the laws pipeline only.

Nova Pro was not reliable enough for final citation selection, so Claude Sonnet
is used here. The model is given the top candidates with their full text and
selects only those directly relevant to the query.

BGE pipeline skips this step and takes top-3 from the Cohere reranker directly.
Docket pipeline also skips this step — Cohere reranker top-3 is the final output.
"""

import json

from config import CLAUDE_MODEL, LAWS_FINAL_TOP_K, load_prompt
from providers import bedrock


def final_llm_selection(
    query: str,
    candidates: list[str],
    client,
    corpus_text_lookup: dict[str, str],
    citation_set: set[str],
    top_k: int = LAWS_FINAL_TOP_K,
) -> list[str]:
    """Ask Claude to select only directly relevant citations from the candidate list.

    Returns fallback top_k if the LLM call fails.
    """
    if not candidates:
        return []

    candidates = candidates[:top_k]

    candidate_lines = [
        f"[{i + 1}] {c}\n{corpus_text_lookup.get(c, '')}"
        for i, c in enumerate(candidates)
        if c in citation_set
    ]
    candidates_block = "\n\n".join(candidate_lines)

    prompt = load_prompt(
        "selection/final_selection.yaml",
        query=query,
        candidates_block=candidates_block,
    )

    raw = bedrock.converse(
        client,
        model_id=CLAUDE_MODEL,
        system_prompt=prompt["system"],
        user_message=prompt["user"],
        max_tokens=1024,
        prefill="[",
    )

    try:
        selected = json.loads("[" + raw)
        return [c for c in selected if c in set(candidates)]
    except Exception:
        return candidates[:top_k]
