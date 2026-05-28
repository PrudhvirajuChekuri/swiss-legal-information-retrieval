"""Stage 0: LLM-based metadata filtering for all three pipelines.

Cascading Metadata Filtering is the key technique in this pipeline.
Before any expensive retrieval (embedding search, FAISS search), the LLM predicts
which slice of the corpus is relevant based on structured metadata:

  - Laws pipeline:   predicts law codes (ZGB, OR, StGB, ...) → filters ~176K laws corpus
  - BGE pipeline:    predicts court divisions (I–V) → filters ~96K BGE corpus
  - Docket pipeline: predicts chambers (1–6) + case type letter → filters ~2.22M docket corpus

This reduces the retrieval space before the first vector search,
dramatically improving precision and reducing cost.
"""

import json
import re

import pandas as pd

import config
from config import (
    ALWAYS_INCLUDE_CODES,
    LAW_CODE_EXPANSIONS,
    NOVA_PRO_MODEL,
    load_prompt,
)
from providers import bedrock


def predict_law_codes(query: str, client, law_codes_json: str) -> list[str]:
    """Predict relevant Swiss law codes for a query (Stage 0, laws pipeline).

    Always includes BV and BGG since they are foundational to any Swiss legal
    question — BV provides constitutional basis, BGG governs Federal Court
    procedure.

    Returns expanded list of law code strings.
    """
    prompt = load_prompt(
        "metadata/predict_law_codes.yaml",
        query=query,
        law_codes_json=law_codes_json,
    )
    raw = bedrock.converse(
        client,
        model_id=NOVA_PRO_MODEL,
        system_prompt=prompt["system"],
        user_message=prompt["user"],
        max_tokens=256,
    )
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group(0)) if match else json.loads(raw)
        predicted = result.get("law_codes", [])
    except Exception:
        predicted = []
    return expand_law_codes(predicted)


def expand_law_codes(predicted_codes: list[str]) -> list[str]:
    """Expand predicted law codes with codependent codes + always-include set.

    Many Swiss statutes only make sense alongside their framework acts or
    implementing regulations (e.g. IVG always cites ATSG). The expansion map
    in config.py encodes these structural dependencies.
    """
    expanded = set(predicted_codes) | ALWAYS_INCLUDE_CODES
    for code in predicted_codes:
        expanded.update(LAW_CODE_EXPANSIONS.get(code, []))
    return list(expanded)


def filter_laws_corpus(laws_df: pd.DataFrame, law_codes: list[str]) -> set[str]:
    """Return the set of law citation strings matching the predicted codes."""
    if not law_codes:
        return set()
    return set(laws_df[laws_df["law_code"].isin(law_codes)]["citation"])


def predict_bge_divisions(query: str, client, bge_divisions_json: str) -> list[str]:
    """Predict relevant BGE court divisions (I–V) for a query (Stage 0, BGE pipeline).

    Falls back to all divisions if prediction fails, to avoid zero recall.
    """
    prompt = load_prompt(
        "metadata/predict_bge_divisions.yaml",
        query=query,
        bge_divisions_json=bge_divisions_json,
    )
    raw = bedrock.converse(
        client,
        model_id=NOVA_PRO_MODEL,
        system_prompt=prompt["system"],
        user_message=prompt["user"],
        max_tokens=128,
    )
    try:
        result = json.loads(raw)
        divisions = [d for d in result.get("bge_divisions", []) if d in {"I", "II", "III", "IV", "V"}]
        return divisions if divisions else ["I", "II", "III", "IV", "V"]
    except Exception:
        return ["I", "II", "III", "IV", "V"]


def filter_bge_corpus(bge_df: pd.DataFrame, divisions: list[str]) -> set[str]:
    """Return the set of BGE citation strings matching the predicted divisions."""
    if not divisions:
        return set()
    return set(bge_df[bge_df["division"].isin(divisions)]["citation"])


def filter_docket_corpus(docket_df: pd.DataFrame, chambers: list[str], procedures: list[str]) -> set[str]:
    """Return the set of docket citation strings matching predicted chambers or case types."""
    if not chambers and not procedures:
        return set()
    mask = pd.Series(False, index=docket_df.index)
    if chambers:
        mask = mask | docket_df["chamber"].astype(str).isin(chambers)
    if procedures:
        mask = mask | docket_df["casetype"].astype(str).str.upper().isin(
            [p.upper() for p in procedures]
        )
    return set(docket_df[mask]["citation"])


def predict_docket_filters(query: str, client, chambers_json: str, procedures_json: str) -> dict:
    """Predict docket chamber numbers and case type letter for a query (Stage 0, docket pipeline).

    Stricter than law/BGE prediction: exactly 1 procedure letter to avoid
    over-filtering on a 2.25M-row corpus where false positives are costly.
    """
    prompt = load_prompt(
        "metadata/predict_docket_filters.yaml",
        query=query,
        chambers_json=chambers_json,
        procedures_json=procedures_json,
    )
    raw = bedrock.converse(
        client,
        model_id=NOVA_PRO_MODEL,
        system_prompt=prompt["system"],
        user_message=prompt["user"],
        max_tokens=128,
    )
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group(0)) if match else json.loads(raw)
        return {
            "chambers": [str(c) for c in result.get("docket_chambers", [])],
            "procedures": result.get("docket_procedures", []),
        }
    except Exception:
        return {"chambers": [], "procedures": []}
