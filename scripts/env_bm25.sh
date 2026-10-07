#!/usr/bin/env bash
set -euo pipefail
BM25_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JAVA_HOME="${BM25_JAVA_HOME:-$BM25_BASE/.runtime/java21}"
export PATH="$JAVA_HOME/bin:$PATH"
export MPLCONFIGDIR="$BM25_BASE/.work/bm25/matplotlib"
BM25_PYTHON="${BM25_PYTHON:-$BM25_BASE/.venv-bm25/bin/python}"
