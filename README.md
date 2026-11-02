# Dhara-Bench

Indian Knowledge Systems evidence-retrieval benchmark: 60,121 corpus chunks, 325 questions, and 7,578 gold evidence links.

Annotation version: `frozen_eval_v3`. Category codes are retained; T06 is named Composite. For T06/T07, `refs.chunk` is the deduplicated union of the current `refs.conjugate_groups` (212 links across 34 questions). Other categories retain their reviewed flat lists. For the 19 Composite (T06) and 15 Multi-hop (T07) questions, `refs.conjugate_groups` additionally records 401 minimal evidence groups (35 singletons). Members within a group are jointly required; alternative groups are valid evidence routes. All 34 questions have at least one group in the Oct23 workbook.

## Files

```text
chunks.json       Complete corpus: id, chunk text, and source ref
Dhara Bench eval set.json        Questions, categories, and gold chunk IDs in refs.chunk
source_metadata.csv        Source descriptions and URLs for the 15 domains
scripts/                   BM25 setup, retrieval, and metric scripts
metric_outputs/            Rankings and per-question, category, and overall scores
README.md
```

[Source metadata](source_metadata.csv) contains the supplied bibliographic descriptions and URLs. Multiple sources or URLs within a domain are separated by line breaks inside the CSV cell.

## Recalculate metrics

From this folder, using Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/pip install numpy==2.2.6 pandas==2.2.3
.venv/bin/python scripts/evaluate_bm25.py
```

This evaluates the included BM25 rankings and saves CSV files in `metric_outputs/`.

Metrics are MRR over the top 20 results, plus precision, recall, F1, and nDCG at 3, 8, 10, and 20. Gold IDs are deduplicated per question. Scores are reported as macro averages over all 325 questions and separately within each of the twelve question categories.

The same command also recalculates `bm25_complete_evidence_per_question.csv` and `bm25_complete_evidence_summary.csv`. CE_Recall@K is 1 when the top K contains every member of at least one group. CE_RR is the reciprocal of the earliest rank at which a whole group is retrieved; CE_MRR averages CE_RR over evaluable questions. Depths 20 and 1000 are reported explicitly; empty group collections are excluded from the CE denominator. Conventional metrics use these current `refs.chunk` lists, including the conjugate unions for T06/T07.

## Run BM25 retrieval again

On Linux x86-64 with Python 3.12:

```bash
bash scripts/setup_bm25.sh
bash scripts/run_all_bm25.sh
```

Setup downloads the pinned dependencies and Java runtime. The BM25 baseline is implemented with Pyserini, which provides a Python interface to the Lucene-based Anserini retrieval toolkit. Retrieval indexes chunk text (`k1=0.9`, `b=0.4`, Porter stemming), returns up to 1,000 chunks per question, and recalculates metrics. This replaces the saved rankings and outputs.

The input JSON files are not modified. Source references are included in each corpus record's `ref` field. Unannotated chunks are not necessarily irrelevant.

Per-question files use `CE_RR@20`, `CE_RR@1000`, and `CE_Recall@K`; summary files use `CE_MRR@20`, `CE_MRR@1000`, and `CE_Recall@K`. The metric values and evaluation cutoffs are unchanged. Complete-set evaluation has precedents in [FEVER](https://aclanthology.org/N18-1074/) and [KILT](https://aclanthology.org/2021.naacl-main.200/). These CE names label the benchmark's explicit chunk-group definitions; CE_RR and CE_MRR adapt reciprocal rank to group completion. KILT's provenance-set Recall@K uses a different aggregation and ranking convention.

The complete-evidence outputs are:

| File in `metric_outputs/` | Contents |
| --- | --- |
| `bm25_complete_evidence_per_question.csv` | Scores, evidence-group counts, and completion rank for each Composite or Multi-hop question |
| `bm25_complete_evidence_summary.csv` | Macro scores and evaluation denominators for Composite, Multi-hop, and both together |

Conventional scores are stored in `per_query_metrics.csv`, `overall_metrics.csv`, and `category_metrics.csv`. Results at each cutoff are provided in `all_metrics_by_cutoff.csv` and `all_category_metrics_by_cutoff.csv`.
