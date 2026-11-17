"""Replay a final policy from the released compact probability CSVs.

Example: python scripts/run_safe_judge.py --dataset covid_reserved --method safe_judge
No model weights, network access or hidden evaluation-label access is needed.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig,run_lara
from src.calibration.safe_judge import SafeJudgeConfig,run_safe_judge
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import file_hash,pair_key,write_json
from scripts.run_error_predictor_ablation import complete_budget

METHODS=('safe_judge','selected_local','lara_1_raw','lara_1_upstream_smooth','lara_topics_raw','lara_topics_upstream_smooth','raw_confidence','random_review')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['dl2019','dl2020','covid_reserved','covid_all'],default='covid_reserved')
    parser.add_argument('--method',choices=METHODS,default='safe_judge')
    parser.add_argument('--human-fraction',type=float,default=.5)
    parser.add_argument('--seed',type=int,default=6000)
    parser.add_argument('--output',type=Path,default=ROOT/'results/reproduction')
    args=parser.parse_args()
    artifact_method='selected_local' if args.method=='safe_judge' else args.method
    if not 0<=args.human_fraction<=1:parser.error('human fraction must be in [0,1]')
    path=ROOT/'artifacts/final'/f'{args.dataset}_predictions.csv'
    with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        r['prob_relevant']=float(r['prob_relevant']);r['true_binary']=int(r['true_binary'])
    pool=[{k:r[k] for k in ('query_id','doc_id','prob_relevant')} for r in rows]
    gold={pair_key(r):r['true_binary'] for r in rows};budget=int(args.human_fraction*len(rows))
    oracle=LabelOracle(gold,budget);p=np.array([r['prob_relevant'] for r in rows]);pred=(p>=.5).astype(int)
    started=time.perf_counter();state=None
    if artifact_method=='selected_local':
        config=json.loads((ROOT/'configs/safe_judge.json').read_text())['review_config']
        human,pred,_,purchases,state=run_safe_judge(pool,oracle,budget,SafeJudgeConfig(**config),args.seed)
    elif args.method.startswith('lara_'):
        grouping='topics' if args.method.startswith('lara_topics') else 1
        prep='upstream_smooth' if args.method.endswith('upstream_smooth') else 'raw'
        scored,purchases,state=run_lara(pool,oracle,budget,LaraConfig(preprocessing=prep),args.seed,grouping)
        by_key={pair_key(r):r for r in scored}
        human=np.array([by_key[pair_key(r)]['label_source']=='human' for r in rows])
        pred=np.array([by_key[pair_key(r)]['predicted_label'] for r in rows])
    else:
        priority=1-np.maximum(p,1-p) if args.method=='raw_confidence' else np.random.default_rng(args.seed).random(len(rows))
        human=complete_budget(pool,[],priority,budget,args.seed)
        purchases=[dict(query_id=r['query_id'],doc_id=r['doc_id'],observed_label=oracle.query(pair_key(r))) for r,h in zip(rows,human) if h]
    bought={pair_key(r) for r in purchases}
    assert len(bought)==len(purchases)==budget==int(human.sum())
    assert bought=={pair_key(r) for r,h in zip(rows,human) if h}
    assert all(r['observed_label']==gold[pair_key(r)] for r in purchases)
    evaluator=PolicyEvaluator(rows);m,q=evaluator.evaluate(human,pred)
    mixed=np.where(human,evaluator.y,pred)
    m.update(mixed_macro_f1=float(f1_score(evaluator.y,mixed,average='macro')),mixed_kappa=float(cohen_kappa_score(evaluator.y,mixed)),
        worst_class_risk=max(m[f'class_{c}_selective_risk'] for c in (0,1)) if m['both_classes_accepted'] else None)
    result=dict(dataset=args.dataset,method=args.method,seed=args.seed,budget=budget,input_sha256=file_hash(path),
        elapsed_seconds=time.perf_counter()-started,metrics=m,per_query=q,purchases=purchases,predicted=pred.tolist(),state=state)
    output=args.output/f'{args.dataset}_{args.method}_{args.seed}_{budget}.json.gz';output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(output,'wt',encoding='utf-8') as f:json.dump(result,f,allow_nan=False)
    if args.dataset!='dl2019':
        with (ROOT/'artifacts/final/policy_runs.csv').open() as f:expected=next((r for r in csv.DictReader(f) if r['dataset']==args.dataset and r['method']==artifact_method and int(r['seed'])==args.seed and int(r['budget'])==budget),None)
        if expected:
            for k in ('automatic_errors','query_mean_auto_error_rate','auto_error_rate','selective_risk','mixed_macro_f1','mixed_kappa'):
                if m[k] is not None:assert np.isclose(m[k],float(expected[k])),k
            result['published_metrics_match']=True
    print(json.dumps(dict(output=str(output),metrics=m,published_metrics_match=result.get('published_metrics_match')),indent=2))


if __name__=='__main__':main()
