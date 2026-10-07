#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_bm25.sh"
# A unique new index avoids accidentally appending to a previous run.
BM25_INDEX="$BM25_BASE/.work/bm25/index-$(date -u +%Y%m%dT%H%M%S)-$$"
printf '%s\n' "$BM25_INDEX" > "$BM25_BASE/.work/bm25/current_index.txt"
"$BM25_PYTHON" -m pyserini.index.lucene \
  --collection JsonCollection --input "$BM25_BASE/.work/bm25/collection" \
  --index "$BM25_INDEX" --generator DefaultLuceneDocumentGenerator \
  --threads 1 --stemmer porter --storePositions --storeDocvectors --storeRaw \
  > "$BM25_BASE/metric_outputs/index_build.log" 2>&1
