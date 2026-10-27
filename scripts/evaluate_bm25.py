#!/usr/bin/env python3
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from bm25_common import *


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
    overall, category, long_all, long_cat = [], [], [], []
    for scope in ['Overall'] + CATEGORIES:
        frame = per if scope == 'Overall' else per[per.category == scope]
        values = frame[columns].to_numpy(dtype=float)
        means = values.mean(axis=0)
        aggregate = {'n_questions':len(frame)}
        for col, mean in zip(columns, means):
            aggregate[col] = float(mean)
        if scope == 'Overall': overall.append(aggregate)
        else: category.append({'category':scope, **aggregate})
        for k in CUTOFFS:
            row = {'k':k, **{NAMES[m]:aggregate[f'{m}@{k}'] for m in METRICS}}
            if scope == 'Overall': long_all.append(row)
            else: long_cat.append({'category':scope, 'n_questions':len(frame), **row})
        print(f'Computed macro metrics: {scope}',flush=True)
    for name, data in [('overall_metrics',overall),('category_metrics',category),
                       ('all_metrics_by_cutoff',long_all),
                       ('all_category_metrics_by_cutoff',long_cat)]:
        pd.DataFrame(data).to_csv(OUT / f'{name}.csv',index=False)
    evaluate_complete_evidence(queries, run)
    current = json.loads((OUT / 'run_metadata.json').read_text())
    if 'retrieval_completed_utc' not in current:
        current['retrieval_completed_utc'] = current.get('completed_utc')
    # Earlier canonical-run checks describe retrieval provenance, not this evaluation.
    for key in ('reproducibility_check', 'final_checks'):
        if key in current:
            current.setdefault('historical_run_checks', {})[key] = current.pop(key)
    dump(OUT / 'run_metadata.json', current)
    metadata_update(status='complete', annotation_version='frozen_eval_v3', evaluation_cutoffs=CUTOFFS, mrr_depth=MRR_DEPTH,
                    metric_columns=columns, evaluated_queries=len(per),
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    existing_corpus_artifacts='Corpus retained without filtering or deduplication; frozen_eval_v3 uses current conjugate unions for T06/T07 conventional gold.',
                    conventional_gold_policy='T06/T07: deduplicated union of current conjugate groups. Other categories: reviewed gold.',
                    no_gold_hit_top10=int((per.gold_hits_at_10 == 0).sum()),
                    no_gold_hit_top100=int((per.gold_hits_at_100 == 0).sum()),
                    rank1_gold_hits=int((per.gold_hits_at_1 == 1).sum()),
                    script_sha256={path.name: sha(path) for path in sorted((BASE/'scripts').iterdir()) if path.suffix in ('.py', '.sh')},
                    metric_output_sha256={path.name: sha(path) for path in sorted(OUT.glob('*.csv'))})
    print(pd.DataFrame(overall).to_string(index=False))
    print(f'Saved metrics to {OUT}')


def evaluate_complete_evidence(queries, run):
    rows = []
    for q in queries:
        if q['category'] not in ('T06', 'T07'):
            continue
        groups = q['conjugate_groups']
        rows.append({'query_id': q['query_id'], 'category': q['category_label'],
                     'question': q['question'], 'evaluable': bool(groups),
                     'group_count': len(groups), 'singleton_groups': sum(len(g) == 1 for g in groups),
                     **complete_scores([hit['id'] for hit in run[q['query_id']]], groups)})
    per = pd.DataFrame(rows)
    columns = ['ce_mrr@20', 'ce_mrr@1000'] + [f'ce_recall@{k}' for k in CUTOFFS]
    summaries = []
    for scope in ['T06+T07', 'T06', 'T07']:
        frame = per if scope == 'T06+T07' else per[per.category.str.startswith(scope)]
        valid = frame[frame.evaluable]
        row = {'scope': scope, 'annotated_questions': len(frame), 'evaluable_questions': len(valid),
               'unevaluable_questions': len(frame)-len(valid)}
        row.update({col: float(valid[col].mean()) if len(valid) else None for col in columns})
        summaries.append(row)
    per.to_csv(OUT/'bm25_complete_evidence_per_question.csv', index=False, na_rep='NA')
    pd.DataFrame(summaries).to_csv(OUT/'bm25_complete_evidence_summary.csv', index=False)
    metadata_update(complete_evidence={'annotated_questions': len(per), 'evaluable_questions': int(per.evaluable.sum()),
                                      'groups': int(per.group_count.sum()), 'singleton_groups': int(per.singleton_groups.sum()),
                                      'mrr_depths': [20, 1000], 'recall_cutoffs': CUTOFFS})


if __name__=='__main__': main()
