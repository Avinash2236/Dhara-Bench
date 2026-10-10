# Dhara-Bench

Indian Knowledge Systems evidence-retrieval benchmark: 60,121 corpus chunks, 325 questions, and 7,813 gold evidence links.

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

Metrics are MRR over the top 20 results, plus precision, recall, F1, and nDCG at 3, 8, 10, and 20. Gold IDs are deduplicated per question. Overall and category scores are macro averages. Bootstrap confidence intervals are also included.

## Run BM25 retrieval again

On Linux x86-64 with Python 3.12:

```bash
bash scripts/setup_bm25.sh
bash scripts/run_all_bm25.sh
```

Setup downloads the pinned dependencies and Java runtime. Retrieval indexes chunk text with Pyserini/Anserini BM25 (`k1=0.9`, `b=0.4`, Porter stemming), returns up to 1,000 chunks per question, and recalculates metrics. This replaces the saved rankings and outputs.

The input JSON files are not modified. Source references are included in each corpus record's `ref` field. Unannotated chunks are not necessarily irrelevant.
