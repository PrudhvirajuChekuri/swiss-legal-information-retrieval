"""End-to-end retrieval pipeline for Swiss legal citation retrieval.

This script is the single entry point for the system. It orchestrates:

  1. Data loading — reads the competition corpora and mapping files
  2. Index building — embeds and indexes each corpus on first run, then loads
     from disk on subsequent runs to avoid re-embedding
  3. Pipeline execution — for each query in val.csv, runs all three pipelines
     (laws → BGE → docket) in sequence, then merges the results
  4. Evaluation — computes per-query F1 and macro-averaged F1 against gold
     citations from val.csv
  5. Output — saves predictions to outputs/val_predictions.csv

Pipeline dependency:
    The BGE pipeline consumes law citations predicted by the laws pipeline
    (used to build the law→BGE bridge candidates). The pipelines must
    therefore run in order: laws first, then BGE, then docket.

Artifact caching:
    Embeddings, FAISS indices, and citation bridges are expensive to build —
    embedding the full laws, BGE, and docket corpora can take significant time
    and incurs Bedrock API costs. Artifacts are saved to indices/ after the
    first build and loaded from disk on subsequent runs. Delete indices/ to
    force a rebuild.

Usage:
    python main.py --build-indices-only             # build cached artifacts
    python main.py                                  # full validation run
    python main.py --subset 1000 --query-id val_001 # smoke test

Arguments:
    --subset N   Subsample each corpus to N rows before building indices.
                 Useful for fast local iteration without full Bedrock costs.
                 When --subset is set, existing full-corpus indices are NOT
                 used — a fresh subset index is built in memory (not saved).
    --query-id ID
                 Run a single validation query, useful for smoke tests.
    --build-indices-only
                 Build/load retrieval artifacts and exit before val queries.
"""

import argparse
import json
import resource

import pandas as pd
from tqdm import tqdm

from config import MAPPINGS_DIR, OUTPUTS_DIR, RAW_DATA_DIR
from evaluation.metrics import macro_f1, parse_citations
from indexing import bge as bge_indexing
from indexing import docket as docket_indexing
from indexing import laws as laws_indexing
from pipelines.bge import run_bge_pipeline
from pipelines.docket import run_docket_pipeline
from pipelines.laws import run_laws_pipeline
from providers import bedrock


def load_mapping_json(filename: str) -> str:
    """Read a mapping CSV from data/mappings/ and return it as a JSON string.

    The JSON string is passed directly into prompt templates so the LLM has
    structured reference data when predicting metadata filters.
    """
    df = pd.read_csv(MAPPINGS_DIR / filename)
    return json.dumps(df.to_dict(orient="records"), indent=2)


def merge_citations(*citation_lists: list[str]) -> list[str]:
    """Merge citation lists from all three pipelines into a single deduplicated list.

    Order is preserved and reflects pipeline confidence: laws citations come
    first (highest signal, LLM-selected), then BGE, then docket.
    """
    seen: set[str] = set()
    merged: list[str] = []
    for cits in citation_lists:
        for c in cits:
            if c not in seen:
                seen.add(c)
                merged.append(c)
    return merged


def _dense_cache_is_valid(index, embeddings, expected_rows: int) -> bool:
    """Return True when cached FAISS index and embeddings match the loaded corpus."""
    return (
        index is not None
        and embeddings is not None
        and len(embeddings) == expected_rows
        and index.ntotal == expected_rows
    )


def _memory_checkpoint(label: str) -> None:
    """Print a lightweight process/system memory checkpoint."""
    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    available_gib = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available_gib = int(line.split()[1]) / 1024**2
                    break
    except OSError:
        pass

    if available_gib is None:
        print(f"[memory] {label}: peak_rss={rss_gib:.2f} GiB")
    else:
        print(f"[memory] {label}: peak_rss={rss_gib:.2f} GiB, system_available={available_gib:.2f} GiB")


def main(
    subset: int | None = None,
    query_id: str | None = None,
    build_indices_only: bool = False,
) -> None:
    """Run the full retrieval pipeline on val.csv and evaluate.

    Args:
        subset: If provided, each corpus is truncated to this many rows before
                indexing. Useful for fast local runs without full Bedrock costs.
                Full-corpus indices on disk are ignored when subset is set.
        query_id: If provided, run only that row from val.csv.
        build_indices_only: Exit after index and bridge setup.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    client = bedrock.get_client()

    # Load mapping JSONs for metadata prediction prompts
    law_codes_json = load_mapping_json("swiss_law_codes.csv")
    bge_divisions_json = load_mapping_json("swiss_bge_divisions.csv")
    chambers_json = load_mapping_json("swiss_docket_chambers.csv")
    procedures_json = load_mapping_json("swiss_docket_procedures.csv")

    # Load corpora
    print("Loading corpora...")
    laws_df = laws_indexing.load_laws_df(subset=subset)
    bge_df = bge_indexing.load_bge_df(subset=subset)
    docket_df = docket_indexing.load_docket_df(subset=subset)
    laws_citation_set = set(laws_df["citation"].astype(str))
    bge_citation_set = set(bge_df["citation"].astype(str))

    print(f"  laws:   {len(laws_df):,} rows")
    print(f"  BGE:    {len(bge_df):,} rows")
    print(f"  docket: {len(docket_df):,} rows")
    _memory_checkpoint("corpus load")

    # Build or load indices
    # When --subset is used, always build fresh indices in memory (never load
    # cached full-corpus indices — row positions would not match the subset df)
    print("\nBuilding/loading indices...")

    if subset is None:
        laws_index = laws_indexing.load_laws_index()
        laws_embeddings = laws_indexing.load_laws_embeddings()
    else:
        laws_index = None
        laws_embeddings = None

    if not _dense_cache_is_valid(laws_index, laws_embeddings, len(laws_df)):
        print("  Building laws FAISS index...")
        laws_embeddings, laws_index = laws_indexing.build_laws_index(laws_df, client)
        if subset is None:
            laws_indexing.save_laws_index(laws_embeddings, laws_index)
    _memory_checkpoint("laws index ready")

    if subset is None:
        bge_index = bge_indexing.load_bge_index()
        bge_embeddings = bge_indexing.load_bge_embeddings()
        bridge = bge_indexing.load_bge_bridge()
    else:
        bge_index = None
        bge_embeddings = None
        bridge = None

    if bge_index is None or bridge is None:
        print("  Building BGE FAISS index + bridge...")
        from retrieval.candidate_generator import build_law_to_court_bridge
        bridge = build_law_to_court_bridge(bge_df, laws_citation_set)
        _memory_checkpoint("BGE bridge build")
        bge_embeddings, bge_index = bge_indexing.build_bge_index(bge_df, client)
        if subset is None:
            bge_indexing.save_bge_index(bge_embeddings, bge_index, bridge)
    elif not _dense_cache_is_valid(bge_index, bge_embeddings, len(bge_df)):
        print("  Rebuilding BGE FAISS index...")
        bge_embeddings, bge_index = bge_indexing.build_bge_index(bge_df, client)
        if subset is None:
            bge_indexing.save_bge_index(bge_embeddings, bge_index, bridge)
    _memory_checkpoint("BGE index ready")

    if subset is None:
        docket_index = None
        docket_embeddings = docket_indexing.load_docket_embeddings()
        docket_bridge = docket_indexing.load_docket_bridge()
    else:
        docket_index = None
        docket_embeddings = None
        docket_bridge = None

    if docket_embeddings is None or len(docket_embeddings) != len(docket_df):
        print("  Building docket embeddings...")
        docket_embeddings = docket_indexing.build_docket_embeddings(docket_df, client)
        if subset is None:
            docket_indexing.save_docket_embeddings(docket_embeddings)
    if docket_bridge is None:
        print("  Building docket bridge...")
        from retrieval.candidate_generator import build_law_to_court_bridge
        docket_bridge = build_law_to_court_bridge(docket_df, laws_citation_set)
        _memory_checkpoint("docket bridge build")
        if subset is None:
            docket_indexing.save_docket_bridge(docket_bridge)
    _memory_checkpoint("docket index and bridge ready")

    if build_indices_only:
        print("\nBuild-indices-only mode complete.")
        return

    docket_citation_set = set(docket_df["citation"].astype(str))

    # Load val queries
    val_df = pd.read_csv(RAW_DATA_DIR / "val.csv")
    if query_id:
        val_df = val_df[val_df["query_id"] == query_id]
        if val_df.empty:
            raise ValueError(f"query_id not found in val.csv: {query_id}")
    gold = {
        row["query_id"]: parse_citations(row["gold_citations"])
        for _, row in val_df.iterrows()
    }

    # Run pipelines
    print(f"\nRunning pipelines on {len(val_df)} val queries...\n")
    predictions: dict[str, list[str]] = {}

    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        qid = row["query_id"]
        query = row["query"]

        law_cits = run_laws_pipeline(
            query=query,
            client=client,
            laws_df=laws_df,
            laws_index=laws_index,
            laws_citation_set=laws_citation_set,
            law_codes_json=law_codes_json,
            laws_embeddings=laws_embeddings,
        )

        bge_cits = run_bge_pipeline(
            query=query,
            law_citations=law_cits,
            client=client,
            bge_df=bge_df,
            bge_index=bge_index,
            bge_citation_set=bge_citation_set,
            bridge=bridge,
            bge_divisions_json=bge_divisions_json,
            bge_embeddings=bge_embeddings,
        )

        docket_cits = run_docket_pipeline(
            query=query,
            law_citations=law_cits,
            client=client,
            docket_df=docket_df,
            docket_index=docket_index,
            docket_citation_set=docket_citation_set,
            bridge=docket_bridge,
            chambers_json=chambers_json,
            procedures_json=procedures_json,
            docket_embeddings=docket_embeddings,
        )

        combined = merge_citations(law_cits, bge_cits, docket_cits)
        predictions[qid] = combined
        print(f"{qid}: {len(law_cits)} laws + {len(bge_cits)} BGE + {len(docket_cits)} docket → {len(combined)} total")

    # Evaluate
    score = macro_f1(predictions, gold)
    print(f"\nMacro F1 on val.csv: {score:.4f}")

    # Save predictions
    out_path = OUTPUTS_DIR / "val_predictions.csv"
    rows = [
        {"query_id": qid, "predicted_citations": ";".join(cits)}
        for qid, cits in predictions.items()
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved predictions → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None, help="Subsample each corpus to N rows")
    parser.add_argument("--query-id", type=str, default=None, help="Run one query_id from val.csv")
    parser.add_argument(
        "--build-indices-only",
        action="store_true",
        help="Stop after loading corpora, building/loading indices, and building/loading bridges",
    )
    args = parser.parse_args()
    main(
        subset=args.subset,
        query_id=args.query_id,
        build_indices_only=args.build_indices_only,
    )
