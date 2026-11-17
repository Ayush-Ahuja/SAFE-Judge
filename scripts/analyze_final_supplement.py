"""Conditional query uncertainty for current COVID class-threshold comparisons."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_final_supplement import COVID
from scripts.compare_qwen_transfer import load_cohort, DETERMINISTIC_METHODS
from src.utils.io import read_gzip_json as unpack
from src.utils.io import file_hash,pair_key,write_csv,write_json


def counts(rows,human,pred):
    y=np.array([r['true_binary'] for r in rows]);out=[]
    for q in sorted({r['query_id'] for r in rows}):
        ix=np.array([r['query_id']==q for r in rows])
        out.append([int((ix & ~human & (pred==c)).sum()) for c in (0,1)]+[int((ix & ~human & (pred==c) & (pred!=y)).sum()) for c in (0,1)])
    return np.array(out)


def main():
    rows=load_cohort(COVID,'trec_covid_locked')
    tensors={};summaries=[]
    for method in ('worst_class','common_paid','raw_confidence','lara_1_raw','lara_1_upstream_smooth','selected_local'):
        reps=[]
        for seed in range(6000,6020):
            if method in ('worst_class','common_paid'):
                saved=unpack(ROOT/'results/experiments/final_supplement_v1/threshold_allocations'/f'{seed}_200_0.5_{method}.json.gz')
                cc=saved['class_counts']
                reps.append(np.array([[r[f'accepted_{c}'] for c in (0,1)]+[r[f'errors_{c}'] for c in (0,1)] for r in cc]))
            else:
                s=6000 if method in DETERMINISTIC_METHODS else 6000+(seed-6000)%10
                saved=unpack(ROOT/'results/experiments/covid_proposal_comparison_v1/allocations/trec_covid_locked'/method/f'seed_{s}_budget_2000.json.gz')
                keys={pair_key(r) for r in saved['purchases']}
                human=np.array([pair_key(r) in keys for r in rows]);pred=np.array(saved['automatic_predicted_labels'])
                reps.append(counts(rows,human,pred))
        tensors[method]=np.array(reps)
    draws=np.random.default_rng(20261117).integers(0,40,(2000,40))
    estimates={}
    for method,cc in tensors.items():
        totals=cc.sum(1);risk=np.max(totals[:,2:]/totals[:,:2],axis=1).mean()
        boot=[]
        for ids in draws:
            x=cc[:,ids,:].sum(1)
            rr=np.divide(x[:,2:],x[:,:2],out=np.full_like(x[:,2:],np.nan,dtype=float),where=x[:,:2]>0)
            boot.append(np.max(rr,axis=1).mean())
        estimates[method]=np.array(boot)
        summaries.append(dict(method=method,worst_class_risk=float(risk)))
    contrasts=[]
    for method in tensors:
        if method=='worst_class':continue
        delta=estimates['worst_class']-estimates[method];delta=delta[np.isfinite(delta)]
        contrasts.append(dict(comparator=method,difference=next(r['worst_class_risk'] for r in summaries if r['method']=='worst_class')-next(r['worst_class_risk'] for r in summaries if r['method']==method),low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)),defined_draws=len(delta),draws=2000,scope='Conditional paired query resampling; average class risk across 20 repetitions, baseline allocations reused modulo ten seeds'))
    write_csv(ROOT/'artifacts/final/covid_threshold_query_intervals.csv',contrasts)
    write_json(ROOT/'artifacts/final/threshold_statistical_validation.json',dict(driver_sha256=file_hash(Path(__file__)),comparisons=len(contrasts),evaluation_labels_enter_selection=False))
    print(contrasts)


if __name__=='__main__':main()
