#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_bm25.sh"
mkdir -p "$BM25_BASE/.work/bm25" "$BM25_BASE/metric_outputs"
"$BM25_PYTHON" "$BM25_BASE/scripts/prepare_pyserini_collection.py"
bash "$BM25_BASE/scripts/build_bm25_index.sh"
"$BM25_PYTHON" "$BM25_BASE/scripts/run_bm25.py"
"$BM25_PYTHON" "$BM25_BASE/scripts/evaluate_bm25.py"
