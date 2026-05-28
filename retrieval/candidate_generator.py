"""Stage 1 candidate generation for laws and court-decision pipelines.

Laws pipeline (Stage 1):
  Extracts explicit citation mentions from the query text, expands to
  paragraph-level (Abs.) citations, and expands by shared title.

  Motivation: queries sometimes name an article explicitly (e.g. "Art. 176 ZGB")
  but gold citations are always at paragraph level (Art. 176 Abs. 1 ZGB). The
  expansion to paragraph level and to same-title articles ensures we capture
  all relevant rows in the laws corpus.

Court-decision pipelines (Stage 1 — bridge):
  Use the law citations predicted by the laws pipeline to find court decisions
  that reference the same articles. This is an intentional cross-pipeline
  dependency: if a case is relevant to OR Art. 97, then decisions that
  discuss OR Art. 97 are likely relevant to the same query.
"""

import json
import re
from collections import defaultdict

import pandas as pd

from config import LAWS_STAGE1_BUDGET, BGE_STAGE1_BUDGET, NOVA_PRO_MODEL, load_prompt
from providers import bedrock


# ---------------------------------------------------------------------------
# Laws Stage 1
# ---------------------------------------------------------------------------

def stage1_laws(
    query: str,
    client,
    laws_df: pd.DataFrame,
    laws_citation_set: set[str],
    budget: int = LAWS_STAGE1_BUDGET,
) -> list[str]:
    """Extract explicit citations from query, expand to Abs. level + by title."""
    raw = _extract_citations(query, client)
    step2 = _expand_to_paragraphs(raw, laws_df)
    step3 = _filter_and_dedupe(step2, laws_citation_set)
    step4 = _expand_by_title(step3, laws_df, budget)
    return step4


def _extract_citations(query: str, client) -> list[str]:
    prompt = load_prompt("retrieval/extract_citations.yaml", query=query)
    raw = bedrock.converse(
        client,
        model_id=NOVA_PRO_MODEL,
        system_prompt=prompt["system"],
        user_message=prompt["user"],
        max_tokens=512,
    )
    try:
        return json.loads(raw)
    except Exception:
        return []


def _expand_to_paragraphs(citations: list[str], laws_df: pd.DataFrame) -> list[str]:
    """Expand article-level citations to all their paragraph-level (Abs.) variants."""
    expanded = []
    for citation in citations:
        tokens = citation.split()
        if len(tokens) < 3:
            continue
        article_prefix = tokens[0] + " " + tokens[1] + " "
        law_code = tokens[-1]
        matches = laws_df.loc[
            laws_df["citation"].apply(
                lambda x: isinstance(x, str)
                and article_prefix in x
                and x.endswith(law_code)
            ),
            "citation",
        ].tolist()
        expanded.extend(matches)
    return expanded


def _filter_and_dedupe(citations: list[str], citation_set: set[str]) -> list[str]:
    seen = set()
    result = []
    for c in citations:
        if c in citation_set and c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _expand_by_title(citations: list[str], laws_df: pd.DataFrame, budget: int) -> list[str]:
    """Expand citation list by adding all citations that share the same law title.

    Swiss law rows with the same title cover the same legal concept across
    different paragraph levels — expanding by title captures all of them.
    """
    citation_to_title = laws_df.set_index("citation")["title"].to_dict()
    title_to_citations = laws_df.groupby("title")["citation"].apply(list).to_dict()

    result = list(citations)
    seen = set(result)
    slots = budget - len(result)

    for citation in citations:
        if slots <= 0:
            break
        title = citation_to_title.get(citation)
        if not title:
            continue
        for c in title_to_citations.get(title, []):
            if c not in seen and slots > 0:
                seen.add(c)
                result.append(c)
                slots -= 1
    return result


# ---------------------------------------------------------------------------
# Court Stage 1 — bridge from law citations to court decisions
# ---------------------------------------------------------------------------

# French/Italian → German law code normalisation for bridge extraction
_CODE_FR_TO_DE = {
    "CO": "OR", "CC": "ZGB", "CCS": "ZGB", "CP": "StGB", "CPC": "ZPO",
    "CPP": "StPO", "LTF": "BGG", "LDIP": "IPRG", "LCD": "UWG",
    "LP": "SchKG", "LEF": "SchKG", "LCR": "SVG", "LAA": "UVG",
    "LAINF": "UVG", "LAVS": "AHVG", "LPGA": "ATSG", "LAI": "IVG",
    "LCart": "KG", "LPM": "MSchG", "LDA": "URG", "LPD": "DSG",
}

_ART_RE = re.compile(
    r"[Aa]rt\.?\s{1,4}(\d+[a-z]*)\s*"
    r"(?:(?:Abs|abs|al|cpv|alin)\.?\s{1,4}(\d+[a-z]*)\s*)?"
    r"(?:(?:lit|lett|Ziff|ch|n)\.?\s{1,4}[a-z0-9]+\s*)*"
    r"([A-Z][A-Za-z]{1,7})\b"
)


def build_law_to_court_bridge(court_df: pd.DataFrame, laws_citation_set: set[str]) -> dict[str, list[str]]:
    """Build offline mapping: law citation → list of court citations that reference it.

    Scans court consideration texts for article references and maps them back
    to their canonical law citation strings. This is expensive to build but
    fast to query at retrieval time.
    """
    bridge: dict[str, list[str]] = defaultdict(list)
    for _, row in court_df.iterrows():
        for law_cit in _extract_law_refs(row["text"], laws_citation_set):
            bridge[law_cit].append(row["citation"])
    return dict(bridge)


def build_law_to_bge_bridge(bge_df: pd.DataFrame, laws_citation_set: set[str]) -> dict[str, list[str]]:
    """Backward-compatible wrapper for the court bridge builder."""
    return build_law_to_court_bridge(bge_df, laws_citation_set)


def _extract_law_refs(text: str, laws_citation_set: set[str]) -> set[str]:
    found = set()
    for m in _ART_RE.finditer(str(text)):
        art_num = m.group(1)
        abs_num = m.group(2)
        de_code = _CODE_FR_TO_DE.get(m.group(3), m.group(3))
        if abs_num:
            cit = f"Art. {art_num} Abs. {abs_num} {de_code}"
            if cit in laws_citation_set:
                found.add(cit)
        cit_art = f"Art. {art_num} {de_code}"
        if cit_art in laws_citation_set:
            found.add(cit_art)
    return found


def stage1_bge(
    law_citations: list[str],
    bridge: dict[str, list[str]],
    bge_df: pd.DataFrame,
    budget: int = BGE_STAGE1_BUDGET,
) -> list[str]:
    """Find BGE candidates via law citation bridge, then expand to full decisions.

    For each matched BGE consideration, all other considerations from the same
    decision are also added — a single decision often has multiple relevant
    sections.
    """
    base_to_considerations = bge_df.groupby("bge_base")["citation"].apply(list).to_dict()

    seen: set[str] = set()
    candidates: list[str] = []

    def add(cit: str) -> bool:
        if len(candidates) >= budget:
            return False
        if cit not in seen:
            seen.add(cit)
            candidates.append(cit)
        return len(candidates) < budget

    bridge_hits: list[str] = []
    for law_cit in law_citations:
        for bge_cit in bridge.get(law_cit, []):
            if bge_cit not in seen:
                bridge_hits.append(bge_cit)
                if not add(bge_cit):
                    return candidates

    for bge_cit in bridge_hits:
        rows = bge_df.loc[bge_df["citation"] == bge_cit, "bge_base"].values
        if len(rows) == 0:
            continue
        for sibling in base_to_considerations.get(rows[0], []):
            if not add(sibling):
                return candidates

    return candidates
