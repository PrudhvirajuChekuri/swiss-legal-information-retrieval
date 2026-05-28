"""Central configuration: paths, model IDs, pipeline constants, prompt loader."""

import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MAPPINGS_DIR = DATA_DIR / "mappings"
RAW_DATA_DIR = DATA_DIR / "raw"
INDICES_DIR = ROOT_DIR / "indices"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PROMPTS_DIR = ROOT_DIR / "prompts"

# ---------------------------------------------------------------------------
# AWS
# ---------------------------------------------------------------------------

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# Model IDs (AWS Bedrock)
# ---------------------------------------------------------------------------

NOVA_PRO_MODEL = "amazon.nova-pro-v1:0"
COHERE_EMBED_MODEL = "cohere.embed-multilingual-v3"
COHERE_RERANK_MODEL = "cohere.rerank-v3-5:0"
CLAUDE_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

EMBED_BATCH_SIZE = 96
EMBED_NUM_WORKERS = 16
EMBED_MAX_CHARS = 2048  # Cohere hard limit

# ---------------------------------------------------------------------------
# Laws pipeline top-K constants
# ---------------------------------------------------------------------------

LAWS_STAGE1_BUDGET = 100
LAWS_DENSE_SUBQ_TOP_K = 10   # top-k per sub-question
LAWS_DENSE_QUERY_TOP_K = 25  # top-k for the original query
LAWS_RERANK_TOP_K = 50
LAWS_FINAL_TOP_K = 15        # fed to Claude for final selection

# ---------------------------------------------------------------------------
# BGE pipeline top-K constants
# ---------------------------------------------------------------------------

BGE_STAGE1_BUDGET = 100
BGE_DENSE_SUBQ_TOP_K = 10
BGE_DENSE_QUERY_TOP_K = 25
BGE_RERANK_TOP_K = 50
BGE_FINAL_TOP_K = 3          # top-3 from reranker, no LLM step

# ---------------------------------------------------------------------------
# Docket pipeline top-K constants
# ---------------------------------------------------------------------------

DOCKET_DENSE_SUBQ_TOP_K = 10
DOCKET_DENSE_QUERY_TOP_K = 25
DOCKET_RERANK_TOP_K = 50
DOCKET_FINAL_TOP_K = 3

# ---------------------------------------------------------------------------
# Laws metadata filtering
# ---------------------------------------------------------------------------

# BV (Federal Constitution) and BGG (Federal Supreme Court Act) are
# foundational to any Swiss legal question — always include them so the
# filter never excludes constitutional or jurisdictional citations.
ALWAYS_INCLUDE_CODES: set[str] = {"BV", "BGG"}

# Codependency map: when code X is predicted, also search codes Y, Z.
# Derived from Swiss law structure — many statutes are only meaningful
# alongside their implementing regulations or parent framework acts.
LAW_CODE_EXPANSIONS: dict[str, list[str]] = {
    # Social insurance — ATSG is the general framework underlying all branches
    "IVG":  ["ATSG", "IVV"],
    "AHVG": ["ATSG", "AHVV"],
    "UVG":  ["ATSG", "UVV"],
    "KVG":  ["ATSG", "KVV"],
    "AVIG": ["ATSG", "AVIV"],
    "BVG":  ["ATSG", "BVV"],
    "ATSG": ["IVG", "AHVG", "UVG", "KVG", "AVIG", "BVG"],

    # Civil law — ZGB and OR frequently co-occur in Swiss civil cases
    "ZGB":  ["OR"],
    "OR":   ["ZGB"],

    # Debt enforcement touches both civil codes and civil procedure
    "SchKG": ["ZGB", "OR", "ZPO"],

    # Private international law
    "IPRG": ["ZGB", "OR", "ZPO"],

    # Criminal — substantive code and procedure always cited together
    "StGB":  ["StPO", "StBOG"],
    "StPO":  ["StGB", "StBOG"],
    "StBOG": ["StGB", "StPO"],
    "MStG":  ["MStP"],
    "MStP":  ["MStG"],

    # Road traffic
    "SVG": ["VRV", "VZV", "VTS"],

    # Immigration
    "AIG":   ["VZAE", "AsylG"],
    "AsylG": ["AIG", "VZAE", "AsylV"],

    # Tax
    "DBG":   ["StHG", "MWSTG"],
    "StHG":  ["DBG"],
    "MWSTG": ["MWSTV"],

    # Financial markets
    "FinfraG": ["FinfraV"],
    "KAG":     ["KKV-FINMA", "KKV"],
}

# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def load_prompt(prompt_path: str | Path, **kwargs: str) -> dict[str, str]:
    """Load a YAML prompt template and format placeholders.

    Returns a dict with 'system' and 'user' keys.
    """
    path = Path(prompt_path)
    if not path.is_absolute():
        path = PROMPTS_DIR / path
    with open(path) as f:
        template = yaml.safe_load(f)
    return {
        "system": template["system"].format(**kwargs) if kwargs else template["system"],
        "user": template["user"].format(**kwargs) if kwargs else template["user"],
    }
