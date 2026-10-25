"""Shared fixed baseline configuration, validation, and metric definitions."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'metric_outputs'
WORK = BASE / '.work/bm25'
EVAL = BASE / 'Dhara Bench eval set.json'
CORPUS = BASE / 'chunks.json'
CUTOFFS = [3, 8, 10, 20]
MRR_DEPTH = 20
CATEGORIES = [f'T{i:02}' for i in range(1, 13)]
EXPECTED = dict(zip(CATEGORIES, [51, 40, 22, 25, 31, 19, 15, 26, 18, 29, 32, 17]))
METRICS = ['precision', 'recall', 'f1', 'ndcg']
NAMES = {'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'ndcg': 'nDCG', 'mrr': 'MRR'}
RUN_NAME = 'dhara_bm25_fixed'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows, columns):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        w = csv.DictWriter(stream, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def integer(value):
    if isinstance(value, bool) or not re.fullmatch(r'\d+(?:\.0+)?', str(value).strip()):
        raise ValueError(f'Invalid integer-like ID: {value!r}')
    return int(str(value).strip().split('.')[0])


def load_inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads(CORPUS.read_text(encoding='utf-8'))
    corpus = {}
    for row in records:
        docid = integer(row['id'])
        if docid in corpus:
            raise ValueError(f'Duplicate corpus ID: {docid}')
        if not isinstance(row.get('chunk'), str) or not isinstance(row.get('ref'), str):
            raise ValueError(f'Invalid chunk/ref: {docid}')
        corpus[docid] = row
    queries = []
    for row in json.loads(EVAL.read_text(encoding='utf-8')):
        qid = integer(row['query_id'])
        question = row['question']
        category = row['category'].split(':', 1)[0].strip()
        original = [integer(i) for i in row['refs']['chunk']]
        gold = set(original)
        if not isinstance(question, str) or not question.strip() or not gold:
            raise ValueError(f'Empty question or gold set: {qid}')
        if gold - corpus.keys():
            raise ValueError(f'Orphan gold IDs for query {qid}: {sorted(gold - corpus.keys())}')
        groups = row['refs'].get('conjugate_groups')
        if category in ('T06', 'T07'):
            validate_groups(groups, corpus)
            if gold != set(conjugate_flat_gold(groups)) or len(original) != len(gold):
                raise ValueError(f'Flat gold must equal the deduplicated conjugate union: {qid}')
        elif groups is not None:
            raise ValueError(f'Unexpected conjugate annotation for {qid}')
        queries.append({'query_id':qid, 'question':question, 'category':category,
                        'category_label':row['category'], 'conjugate_groups':groups,
                        'gold_ids':list(dict.fromkeys(original)), 'gold_set':gold})
    if len(queries) != 325 or len({q['query_id'] for q in queries}) != 325:
        raise ValueError('Expected 325 unique query IDs')
    if Counter(q['category'] for q in queries) != Counter(EXPECTED):
        raise ValueError('Frozen category counts do not match')
    return sorted(queries, key=lambda q:q['query_id']), corpus


def conjugate_flat_gold(groups):
    return list(dict.fromkeys(docid for group in groups for docid in group))


def validate_groups(groups, corpus):
    if not isinstance(groups, list):
        raise ValueError('Expected an explicit list of conjugate groups')
    seen = set()
    for group in groups:
        if not isinstance(group, list) or not group or any(type(i) is not int for i in group):
            raise ValueError('Each group must be a nonempty integer list')
        key = frozenset(group)
        if len(key) != len(group) or key in seen or key - corpus.keys():
            raise ValueError('Duplicate group/member or orphan conjugate ID')
        seen.add(key)
    if any(a < b for a in seen for b in seen):
        raise ValueError('Conjugate groups must be inclusion-minimal')


def complete_scores(ranking, groups):
    if len(ranking) != len(set(ranking)):
        raise ValueError('Duplicate retrieved IDs')
    ranks = {docid: rank for rank, docid in enumerate(ranking[:1000], 1)}
    completion = min((max(ranks[i] for i in group) for group in groups
                      if all(i in ranks for i in group)), default=None)
    return {'completion_rank': completion,
            **{f'ce_mrr@{k}': (1 / completion if completion is not None and completion <= k else 0.)
               if groups else None for k in (20, 1000)},
            **{f'ce_recall@{k}': int(completion is not None and completion <= k)
               if groups else None for k in CUTOFFS}}


def metric_row(ranked_ids, gold):
    if not gold or len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError('Gold must be nonempty and retrieved IDs unique')
    first = next((i for i, docid in enumerate(ranked_ids, 1) if docid in gold), None)
    # Same binary metrics and top-20 MRR convention as metrics_updated's
    # report_rerank_metrics.scores. Keep deeper hits only as diagnostics.
    out = {'rank_first_gold_hit': first,
           'mrr': 1 / first if first is not None and first <= MRR_DEPTH else 0.0}
    for k in [1, 100]:
        out[f'gold_hits_at_{k}'] = sum(docid in gold for docid in ranked_ids[:k])
    for k in CUTOFFS:
        relevance = [int(docid in gold) for docid in ranked_ids[:k]]
        hits = sum(relevance)
        precision, recall = hits / k, hits / len(gold)
        dcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(relevance, 1))
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
        out.update({f'gold_hits_at_{k}': hits, f'precision@{k}': precision, f'recall@{k}': recall,
                    f'f1@{k}': 2 * precision * recall / (precision + recall) if hits else 0.0,
                    f'ndcg@{k}': dcg / ideal})
    return out


def metadata_update(**kwargs):
    path = OUT / 'run_metadata.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(kwargs)
    dump(path, data)
    return data
