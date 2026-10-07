#!/usr/bin/env python3
from collections import defaultdict
import numpy as np
import pandas as pd
from bm25_common import *


def bootstrap(values):
    rng = np.random.default_rng(42)
    n = len(values)
    means = np.empty((10000, values.shape[1]), dtype=float)
    for begin in range(0, 10000, 100):
        sampled = rng.integers(0, n, size=(100, n))
        means[begin:begin+100] = values[sampled].mean(axis=1)
    return np.quantile(means, [.025, .975], axis=0, method='linear')


def main():
    queries, corpus = load_inputs()
    metadata = json.loads((OUT / 'run_metadata.json').read_text())
    if (sha(EVAL) != metadata['evaluation_json_sha256'] or
            sha(CORPUS) != metadata['corpus_file_sha256'] or
            sha(OUT / 'run_bm25_top1000.trec') != metadata['run_sha256']):
        raise ValueError('BM25 input/run checksum mismatch')
    metadata_update(status='evaluating')
    qids = {q['query_id'] for q in queries}
    run = defaultdict(list)
    for line in (OUT / 'run_bm25_top1000.trec').read_text().splitlines():
        qid, q0, docid, rank, score, name = line.split()
        qid, docid, rank, score = int(qid), int(docid), int(rank), float(score)
        assert q0 == 'Q0' and name == RUN_NAME and qid in qids and docid in corpus and math.isfinite(score)
        assert rank == len(run[qid]) + 1
        if run[qid]: assert score <= run[qid][-1]['score']
        run[qid].append({'id':docid, 'rank':rank, 'score':score})
    assert set(run) == qids
    per_query = []
    for q in queries:
        ranked = [hit['id'] for hit in run[q['query_id']]]
        assert 0 < len(ranked) <= 1000 and len(ranked) == len(set(ranked))
        per_query.append({'query_id':q['query_id'],'category':q['category'],'question':q['question'],
                          'number_of_gold_chunks':len(q['gold_set']), **metric_row(ranked, q['gold_set'])})
    per = pd.DataFrame(per_query)
    columns = ['mrr'] + [f'{metric}@{k}' for metric in METRICS for k in CUTOFFS]
    assert len(per) == 325 and per.query_id.nunique() == 325
    assert np.isfinite(per[columns].values).all()
    assert ((per[columns].values >= -1e-12) & (per[columns].values <= 1+1e-12)).all()
    assert (np.diff(per[[f'recall@{k}' for k in CUTOFFS]].values,axis=1) >= -1e-12).all()
    assert (np.diff(per[[f'gold_hits_at_{k}' for k in CUTOFFS]].values,axis=1) >= 0).all()
    per.to_csv(OUT / 'per_query_metrics.csv', index=False, na_rep='NA')
    overall, category, overall_ci, category_ci, long_all, long_cat = [], [], [], [], [], []
    for scope in ['Overall'] + CATEGORIES:
        frame = per if scope == 'Overall' else per[per.category == scope]
        values = frame[columns].to_numpy(dtype=float)
        means = values.mean(axis=0)
        low, high = bootstrap(values)
        aggregate = {'n_questions':len(frame)}
        for col, mean in zip(columns, means):
            aggregate[col] = float(mean)
        if scope == 'Overall': overall.append(aggregate)
        else: category.append({'category':scope, **aggregate})
        for i, col in enumerate(columns):
            metric, k = col.split('@') if '@' in col else ('mrr', str(MRR_DEPTH))
            row = {'metric':NAMES[metric], 'k':int(k), 'value':float(means[i]),
                   'ci_lower':float(low[i]), 'ci_upper':float(high[i])}
            if scope == 'Overall': overall_ci.append(row)
            else: category_ci.append({'category':scope, **row})
        for k in CUTOFFS:
            row = {'k':k, **{NAMES[m]:aggregate[f'{m}@{k}'] for m in METRICS}}
            if scope == 'Overall': long_all.append(row)
            else: long_cat.append({'category':scope, 'n_questions':len(frame), **row})
        print(f'Computed macro metrics and 10,000 question bootstrap resamples: {scope}',flush=True)
    for name, data in [('overall_metrics',overall),('category_metrics',category),('overall_bootstrap_ci',overall_ci),
                       ('category_bootstrap_ci',category_ci),('all_metrics_by_cutoff',long_all),
                       ('all_category_metrics_by_cutoff',long_cat)]:
        pd.DataFrame(data).to_csv(OUT / f'{name}.csv',index=False)
    metadata_update(status='complete', evaluation_cutoffs=CUTOFFS, mrr_depth=MRR_DEPTH,
                    metric_columns=columns, evaluated_queries=len(per))
    print(pd.DataFrame(overall).to_string(index=False))
    print(f'Saved metrics to {OUT}')


if __name__=='__main__': main()
