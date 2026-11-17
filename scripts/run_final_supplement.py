"""Final CPU checks on frozen scores: current COVID thresholds and calibration.

No evaluation labels enter threshold selection or probability calibration.
All calibration pairs become paid human labels; each ledger is checked.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_qwen_transfer import load_cohort, DETERMINISTIC_METHODS
from scripts.run_class_threshold_experiment import sample_calibration
from src.baselines.active_learning import LabelOracle
from src.baselines.lara import LaraConfig, run_lara
from src.calibration.class_thresholds import select_class_thresholds
from src.calibration.diagnostic_review import ReviewConfig, run_diagnostic_review
from src.evaluation.metrics import classification_metrics
from src.evaluation.review_metrics import PolicyEvaluator
from src.utils.io import file_hash, pair_key, read_predictions, write_csv, write_json

COVID = ROOT / 'results/transfer/qwen35_covid_proposal_v1_dataset_full'
OUT = ROOT / 'results/experiments/final_supplement_v1'
SEEDS = list(range(6000, 6010))
POLICY = ROOT / 'configs/diagnostic_review_selected.json'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with gzip.open(temp, 'wt', encoding='utf-8') as f:
        json.dump(value, f, allow_nan=False)
    temp.replace(path)


def metrics(rows, human, predicted):
    ev = PolicyEvaluator(rows)
    m, q = ev.evaluate(human, predicted)
    mixed = np.where(human, ev.y, predicted)
    m['mixed_macro_f1'] = float(f1_score(ev.y, mixed, average='macro'))
    m['mixed_kappa'] = float(cohen_kappa_score(ev.y, mixed))
    m['worst_class_risk'] = max(m[f'class_{c}_selective_risk'] for c in (0, 1)) if m['both_classes_accepted'] else None
    for c in (0, 1):
        ix = ev.raw == c
        m[f'raw_class_{c}_coverage'] = float((~human & ix).sum()/ix.sum())
    return m, q


def allocate(rows, budget, method, seed, config):
    pool = [{k:r[k] for k in ('query_id','doc_id','prob_relevant')} for r in rows]
    oracle = LabelOracle({pair_key(r):int(r['true_binary']) for r in rows}, budget)
    start = time.perf_counter()
    if method == 'selected_local':
        human, pred, _, bought, state = run_diagnostic_review(pool, oracle, budget, ReviewConfig(**config), seed)
    else:
        grouping = 'topics' if method.startswith('lara_topics') else 1
        prep = 'upstream_smooth' if method.endswith('upstream_smooth') else 'raw'
        scored, bought, state = run_lara(pool, oracle, budget, LaraConfig(preprocessing=prep), seed, grouping)
        human = np.array([r['label_source']=='human' for r in scored])
        pred = np.array([r['predicted_label'] for r in scored])
    keys = {pair_key(r) for r in bought}
    assert len(keys) == len(bought) == budget == int(human.sum())
    assert keys == {pair_key(r) for r,h in zip(rows,human) if h}
    return human, pred, bought, state, time.perf_counter()-start


def calibration_job(cost, seed, method, view):
    path = OUT / 'calibration_allocations' / f'{cost}_{seed}_{method}_{view}.json.gz'
    if path.exists():
        with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)['record']
    rows = load_cohort(COVID, 'trec_covid_all')
    info = json.loads((COVID/'inputs/trec_covid.metadata.json').read_text())
    dev_ids = set(info['sampling']['development_query_ids'])
    dev = [r for r in rows if r['query_id'] in dev_ids]
    cal = sample_calibration(dev, cost//10, seed)
    purchased = {pair_key(r) for r in cal}
    remaining = [r for r in rows if pair_key(r) not in purchased]
    test = [r for r in rows if r['query_id'] not in dev_ids]
    params = None
    probability_metrics = None
    if view == 'platt':
        x = logit(np.clip([r['prob_relevant'] for r in cal],1e-12,1-1e-12))[:,None]
        model = LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,tol=1e-9).fit(x,[r['true_binary'] for r in cal])
        params = dict(slope=float(model.coef_[0,0]),intercept=float(model.intercept_[0]))
        def transform(group):
            p=expit(params['slope']*logit(np.clip([r['prob_relevant'] for r in group],1e-12,1-1e-12))+params['intercept'])
            return [{**r,'prob_relevant':float(v)} for r,v in zip(group,p)]
        remaining = transform(remaining)
        test = transform(test)
    probability_metrics = classification_metrics(test)
    probability_metrics.pop('reliability')
    config=json.loads(POLICY.read_text())['review_config']
    human, pred, bought, state, elapsed = allocate(remaining,2500-cost,method,seed,config)
    by_key={pair_key(r):(bool(h),int(p)) for r,h,p in zip(remaining,human,pred)}
    full_human=np.array([pair_key(r) in purchased or by_key[pair_key(r)][0] for r in rows])
    full_pred=np.array([int(r['prob_relevant']>=.5) if pair_key(r) in purchased else by_key[pair_key(r)][1] for r in rows])
    ledger=[dict(query_id=r['query_id'],doc_id=r['doc_id'],observed_label=int(r['true_binary']),stage='target_calibration') for r in cal]+bought
    assert len({pair_key(r) for r in ledger}) == len(ledger) == int(full_human.sum()) == 2500
    gold={pair_key(r):r['true_binary'] for r in rows}
    assert all(r['observed_label']==gold[pair_key(r)] for r in ledger)
    m,q=metrics(rows,full_human,full_pred)
    record=dict(study='target_calibration',dataset='covid_all',calibration_labels=cost,seed=seed,method=method,score_view=view,elapsed_seconds=elapsed,**m)
    save(path,dict(record=record,calibration_parameters=params,heldout_probability_metrics=probability_metrics,purchases=ledger,predicted=full_pred.tolist(),per_query=q,state=state))
    return record


def thresholds():
    allrows=load_cohort(COVID,'trec_covid_all')
    rows=load_cohort(COVID,'trec_covid_locked')
    info=json.loads((COVID/'inputs/trec_covid.metadata.json').read_text())
    ids=set(info['sampling']['development_query_ids'])
    dev=[r for r in allrows if r['query_id'] in ids]
    p=np.array([r['prob_relevant'] for r in rows]);raw=(p>=.5).astype(int)
    records=[]
    for seed in range(6000,6020):
        for cost in (100,200,500):
            cal=sample_calibration(dev,cost//10,seed)
            for fraction in (.25,.5,.75):
                budget=int(fraction*len(rows))
                for objective in ('worst_class','total_error','common_paid'):
                    if objective=='common_paid':
                        tie=np.random.default_rng(seed).permutation(len(rows))
                        order=np.lexsort((tie,np.maximum(p,1-p)))
                        human=np.zeros(len(rows),dtype=bool);human[order[:budget-cost]]=True
                        state={'objective':objective}
                    else:
                        human,state=select_class_thresholds([r['prob_relevant'] for r in cal],[r['true_binary'] for r in cal],p,len(rows)-(budget-cost),seed=seed,objective=objective)
                    m,q=metrics(rows,human,raw)
                    record=dict(study='separate_thresholds',dataset='covid_reserved',seed=seed,calibration_labels=cost,fraction=fraction,method=objective,total_human_labels=cost+int(human.sum()),**m)
                    assert record['total_human_labels']==budget
                    path=OUT/'threshold_allocations'/f'{seed}_{cost}_{fraction}_{objective}.json.gz'
                    ledger=[dict(query_id=r['query_id'],doc_id=r['doc_id'],observed_label=int(r['true_binary']),stage='calibration') for r in cal]
                    ledger += [dict(query_id=r['query_id'],doc_id=r['doc_id'],observed_label=int(r['true_binary']),stage='review') for r,h in zip(rows,human) if h]
                    assert len(ledger)==len({pair_key(r) for r in ledger})==budget
                    counts=[]
                    for qid in sorted({r['query_id'] for r in rows}):
                        ix=np.array([r['query_id']==qid for r in rows])
                        counts.append(dict(query_id=qid,**{f'accepted_{c}':int((ix & ~human & (raw==c)).sum()) for c in (0,1)},**{f'errors_{c}':int((ix & ~human & (raw==c) & (raw != np.array([r['true_binary'] for r in rows]))).sum()) for c in (0,1)}))
                    save(path,dict(record=record,state=state,purchases=ledger,per_query=q,class_counts=counts))
                    records.append(record)
    return records


def common_baselines():
    rows,_=read_predictions(ROOT/'results/stage2/dl2019_qwen35_full_v1/predictions.csv',2)
    split=json.loads((ROOT/'results/baselines/cross_encoder/dl2019_shared_test_v1/split.json').read_text())
    ids=set(split['test_query_ids']); rows=sorted([r for r in rows if r['query_id'] in ids],key=pair_key)
    assert len(rows)==7328
    config=json.loads(POLICY.read_text())['review_config']
    records=[]
    for seed in (42,43,44):
        path=OUT/'common_baseline_allocations'/f'local_{seed}.json.gz'
        if path.exists():
            with gzip.open(path,'rt') as f:saved=json.load(f)
            records.append(saved['record']);continue
        h,p,ledger,state,elapsed=allocate(rows,3664,'selected_local',seed,config)
        m,q=metrics(rows,h,p)
        record=dict(study='common_dl2019',dataset='dl2019_shared33',seed=seed,method='selected_local',calibration_labels=0,total_human_labels=3664,elapsed_seconds=elapsed,**m)
        save(path,dict(record=record,purchases=ledger,predicted=p.tolist(),per_query=q,state=state))
        records.append(record)
    return records


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[POLICY,ROOT/'src/calibration/class_thresholds.py',ROOT/'src/calibration/diagnostic_review.py',ROOT/'src/baselines/lara.py']
    design=dict(source_sha256={p.relative_to(ROOT).as_posix():file_hash(p) for p in sources},
        inputs={str(p.relative_to(ROOT)):file_hash(p) for p in (COVID/'runs/trec_covid/predictions.csv',COVID/'inputs/trec_covid.metadata.json')},
        target_calibration='1000 or 200 labels on the same ten development topics; full-pool total budget 2500; fixed C=1 Platt, no hyperparameter search',
        thresholds='20 seeds, 100/200/500 calibration labels, 25/50/75 percent total cost on 4000 reserved pairs; fixed settings inherited from class_threshold_experiment.json')
    path=OUT/'design.json'
    if path.exists() and json.loads(path.read_text()) != design:raise ValueError('Frozen final-supplement design changed')
    write_json(path,design)
    records=thresholds();print('Threshold rows',len(records),flush=True)
    with ProcessPoolExecutor(max_workers=2) as ex:
        jobs=[ex.submit(calibration_job,c,s,m,v) for c in (200,1000) for s in SEEDS for m in ('selected_local','lara_1_raw') for v in ('raw','platt')]
        for job in as_completed(jobs):
            r=job.result();records.append(r)
            print('Calibration',r['calibration_labels'],r['seed'],r['method'],r['score_view'],flush=True)
    records+=common_baselines()
    write_csv(OUT/'runs.csv',records, list(dict.fromkeys(k for r in records for k in r)))
    write_json(OUT/'experiment.json',dict(status='complete',rows=len(records),design_sha256=file_hash(path),driver_sha256=file_hash(Path(__file__)),unique_ledgers_checked=True,scope='Exploratory CPU evaluation on previously inspected benchmarks; gold oracle, no new inference.'))
    print('Final supplement complete',len(records),flush=True)


if __name__=='__main__':main()
