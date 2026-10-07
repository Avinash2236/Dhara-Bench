#!/usr/bin/env python3
import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from bm25_common import *

metadata_update(status='validating')
queries, corpus = load_inputs()
WORK.mkdir(parents=True, exist_ok=True)
collection = WORK / 'collection'
collection.mkdir(exist_ok=True)
with (collection / 'corpus.jsonl').open('w', encoding='utf-8') as stream:
    for docid, row in sorted(corpus.items()):
        stream.write(json.dumps({'id': str(docid), 'contents': row['chunk']}, ensure_ascii=False) + '\n')
git = subprocess.run(['git', '-C', str(BASE), 'rev-parse', 'HEAD'], capture_output=True, text=True)
metadata_update(status='prepared', started_utc=datetime.now(timezone.utc).isoformat(),
                evaluation_json_sha256=sha(EVAL), corpus_file_sha256=sha(CORPUS),
                collection_sha256=sha(collection / 'corpus.jsonl'),
                python_version=platform.python_version(), pyserini_version=importlib.metadata.version('pyserini'),
                git_commit=git.stdout.strip() if git.returncode == 0 else None,
                bm25_k1=0.9, bm25_b=0.4, retrieval_depth=1000, evaluation_cutoffs=CUTOFFS, mrr_depth=MRR_DEPTH,
                number_of_queries=len(queries), number_of_corpus_chunks=len(corpus),
                query_sanitization='None: exact workbook string passed to LuceneSearcher.search; bag-of-words query generator, not query-parser syntax.',
                indexed_content='Exact chunk text only; canonical stringified corpus ID is a document identifier, not a retrieval term. No ref or source metadata.',
                bootstrap={'resamples':10000,'seed':42,'unit':'question','interval':'percentile 2.5/97.5, NumPy linear quantiles', 'rng':'numpy.random.default_rng(42), reset for each scope; batches of 100'},
                relevance_interpretation='Binary relevance relative to declared expert gold judgments. Unannotated retrieved text is not necessarily semantically irrelevant.',
                parameter_selection='Fixed a priori; no evaluation-driven tuning.',
                existing_corpus_artifacts='Corpus retained without filtering or deduplication; frozen gold links exclude IDs 33577 and 34763 from query 251.')
print(f'Validated {len(queries)} questions; wrote {len(corpus)} exact-content JSONL documents.')
