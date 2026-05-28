"""Citation-level evaluation metrics."""


def compute_f1(predicted: list[str], gold: list[str]) -> float:
    """Compute F1 between a predicted citation set and a gold citation set."""
    pred_set = set(predicted)
    gold_set = set(gold)
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(predictions: dict[str, list[str]], gold: dict[str, list[str]]) -> float:
    """Compute macro-averaged F1 across all query IDs present in gold."""
    scores = [
        compute_f1(predictions.get(qid, []), gold_cits)
        for qid, gold_cits in gold.items()
    ]
    return sum(scores) / len(scores) if scores else 0.0


def parse_citations(citation_str: str) -> list[str]:
    """Split a semicolon-separated citation string into a clean list."""
    return [c.strip() for c in str(citation_str).split(";") if c.strip()]
