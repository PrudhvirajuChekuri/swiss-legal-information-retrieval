# LLM Agentic Legal Information Retrieval — Data Page

**Competition:** LLM Agentic Legal Information Retrieval  
**Host:** Ari Jordan  
**Type:** Community Prediction Competition  
**Status:** Closed
**Goal:** Retrieve the right Swiss legal sources for each question and optimize citation-level F1 on a hidden test set.

## Dataset Description

This page describes the provided data files and their schemas.

If you are building a retriever, the typical workflow is:

1. Learn from `train.csv`.
2. Sanity-check on `val.csv`.
3. Retrieve from `laws_de.csv` and `court_considerations.csv`.
4. Generate `submission.csv` for `test.csv`.

## Files

| File | Description |
|---|---|
| `train.csv` | Public training queries with gold citation labels separated by semicolons. These queries are non-English and are based on LEXam. |
| `val.csv` | Small validation set with 10 English queries and gold citations. It does not match the train distribution. |
| `test.csv` | Test queries only, with no labels. Your notebook must generate predictions for these `query_id`s. It matches the validation distribution and contains 40 English samples: 20 scored on the public leaderboard and 20 on the private leaderboard. |
| `HIDDEN.csv` | Additional test data used to verify potential cheating concerns, such as annotating test data and training on it. |
| `laws_de.csv` | Retrieval corpus of Swiss federal law snippets in German, keyed by canonical citation strings. |
| `court_considerations.csv` | Large retrieval corpus of Swiss Federal Court decision considerations in German, French, or Italian, keyed by canonical citation strings. It includes leading and non-leading decisions going back about 30 years. Older decisions are missing and are not expected in the gold citations. Completeness is not guaranteed. |
| `sample_submission.csv` | Example of the required submission format. |

The competition recommends downloading these files into the `data` directory of the Starter Repo.

## LEXam (CC BY 4.0)

The competition dataset includes material in `train.csv` adapted from:

**LEXam: Benchmarking Legal Reasoning on 340 Law Exams**  
Yu Fan, Jingwei Ni, Jakob Merane, Yang Tian, Yoan Hermstrüwer, Yinya Huang, Mubashara Akhtar, Etienne Salimbeni, Florian Geering, Oliver Dreyer, Daniel Brunner, Markus Leippold, Mrinmaya Sachan, Alexander Stremitzer, Christoph Engel, Elliott Ash, Joel Niklaus (2025).

- **Source:** https://huggingface.co/datasets/LEXam-Benchmark/LEXam/viewer/open_question
- **Paper:** https://arxiv.org/abs/2505.12864
- **License:** CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/
- **Modifications:** Citations were extracted from the `answer` field, most columns were removed, and a subset of row items was selected.
- **No endorsement implied.**

## Swiss Legal Sources (Retrieval Corpus)

The competition includes excerpts of Swiss federal enactments and court decision texts in:

- `laws_de.csv`
- `court_considerations.csv`

These sources are taken from official publications, such as Fedlex and the Swiss Federal Supreme Court’s official publication systems.

Under Swiss law, official enactments and official decisions or reports of authorities are not protected by copyright under the Swiss Copyright Act, Art. 5.

The excerpts are provided for research and competition purposes. There is no guarantee of completeness or officialness.

## Schemas

### `train.csv` / `val.csv`

| Column | Type | Description |
|---|---|---|
| `query_id` | string | Unique ID, such as `train_0123` or `val_0007`. |
| `query` | string | Legal question in English. |
| `gold_citations` | string | Ground-truth citations separated by semicolons, such as `Art. 11 Abs. 2 OR;BGE 119 II 449 E. 3.4`. |

### `test.csv`

| Column | Type | Description |
|---|---|---|
| `query_id` | string | Unique ID, such as `test_0045`. |
| `query` | string | Legal question in English. |

### `laws_de.csv`

| Column | Type | Description |
|---|---|---|
| `citation` | string | Canonical identifier for the law snippet. Use this as the ID you predict. |
| `text` | string | Full German text for that law chunk. |

### `court_considerations.csv`

| Column | Type | Description |
|---|---|---|
| `citation` | string | Canonical identifier for the decision or consideration. Use this as the ID you predict. |
| `text` | string | Consideration text in German, French, or Italian. |

## Submission Format

Your notebook must write a `submission.csv` with:

- One row per `query_id` in `test.csv`.
- Semicolon-separated citations in `predicted_citations`.
- An empty string is allowed if you predict no citations.

```csv
query_id,predicted_citations
test_001,"Art. 11 Abs. 2 OR;BGE 139 I 2 E. 3.1"
test_002,"5A_800/2019 E 5."
test_003,""
```

## Retrieval Granularity

Each row in `laws_de.csv` and `court_considerations.csv` represents:

- One retrievable unit.
- One valid prediction ID.

Even if multiple rows originate from the same decision, such as `BGE 139 I 2`, they are treated as distinct citations when their canonical citation strings differ.

In the beginning, it is reasonable to treat citations as opaque, domain-specific IDs. You do not need to parse them to compete, but some understanding of their structure can help optimize your solution.

## Citation Identifiers

You will see multiple citation families in the labels and corpora.

### Federal Law Citations

Article-level examples:

```text
Art. 1 ZGB
Art. 117 StGB
```

Paragraph-level examples:

```text
Art. 11 Abs. 2 OR
Art. 45 Abs. 2 AHVG
```

For articles split into paragraphs, the gold citations are the relevant paragraph citations, not full article citations.

For example, if `Art. 11 Abs. 2 OR` exists, then `Art. 11 OR` cannot be a valid gold citation.

### Federal Court Decisions

Leading decisions may appear as:

```text
BGE 116 Ia 56 E 1.
BGE 121 III 38 E. 2b
BGE 145 II 32 E. 3.1
```

Non-leading but still public decisions can appear as docket-style identifiers:

```text
5A_800/2019 E 2.
2C_123/2020 E 1.2.3
```

For scoring, all of these are just citation IDs. Your goal is to output the correct set per query.

## Data FAQ

### What does “ground truth” mean here?

Legal research does not have a single, stable answer set. As the saying goes, “if you want three opinions ask two lawyers.” Disagreement is a feature of the domain, not just label noise.

The competition references the paper **Legal Disagreement**:  
https://www.cambridge.org/core/services/aop-cambridge-core/content/view/9D4C5757ED50A48B1FBA759563C48DEC/S089765462200065Xa.pdf/legal-disagreement.pdf

Implications:

- Gold citations are best treated as high-quality references, though they are not infallible.
- The organizers have observed cases where they disagree with the underlying data or interpretation.
- You should expect that it is impossible to predict the labels perfectly.
- Part of the task is to predict the opinion of the domain expert who annotated the data, rather than what a “perfect judge” would select as citations.
- You can also consider this label noise.

### Why are the test set and especially the validation set small?

Reasons given by the competition:

- It is expensive to annotate high-quality legal data.
- It may be slow for participants to do inference on many samples for this kind of task.
- You can consider it a few-shot transfer-learning task.

### Can I predict citations that are not in the corpus?

No. Citations outside the corpus will not match anything and will be scored as false positives.

Treat the corpus citation strings as the closed vocabulary of valid outputs.

### Are queries always in English? Are sources always in German?

Queries are in English.

Sources are mostly German, but some court considerations may be in French or Italian.

Your system should be robust to cross-lingual retrieval.
