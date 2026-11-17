"""Generate publication figures and LaTeX tables from curated final CSVs.

This step is CPU-only and independent of model weights and private caches.
The narrative is maintained in reports/final/{paper,supplement}.tex.
"""
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.io import file_hash, write_json

DATA=ROOT/'artifacts/final'; REPORT=ROOT/'reports/final'; FIG=REPORT/'figures'; TAB=REPORT/'tables'
NAMES={'selected_local':'SAFE-Judge','local_unguarded':'SAFE-Judge','lara_1_raw':'LARA G/raw','lara_1_upstream_smooth':'LARA G/norm',
 'lara_topics_raw':'LARA Q/raw','lara_topics_upstream_smooth':'LARA Q/norm','raw_confidence':'Confidence','random_review':'Random'}
METHODS=('selected_local','lara_1_raw','lara_1_upstream_smooth','lara_topics_raw','lara_topics_upstream_smooth','raw_confidence','random_review')
COLORS=['#0072B2','#D55E00','#E69F00','#009E73','#CC79A7','#555555','#999999']
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','axes.axisbelow':True})


def read(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def val(row,key):
    v=row.get(key,'');return float(v) if v not in ('',None) else None


def fmt(value,percent=False,digits=3):
    if value is None:return '--'
    number=Decimal(str(round(value,12)))*(100 if percent else 1)
    return str(number.quantize(Decimal(1).scaleb(-digits),rounding=ROUND_HALF_UP))


def mean(rows,key):
    x=[val(r,key) for r in rows if val(r,key) is not None]
    return float(np.mean(x)) if x else None


def tex(name,headers,rows,align=None,long=False,best=None,groups=None):
    rows=[list(row) for row in rows]
    grouped_rows=defaultdict(list)
    if groups is None:
        groups=[0]*len(rows)
    if len(groups)!=len(rows):
        raise ValueError('Table groups must match the rows')
    for row,group in zip(rows,groups):
        if group is not None:
            grouped_rows[group].append(row)
    for group_rows in grouped_rows.values():
        for col,direction in (best or {}).items():
            values=[float(row[col]) for row in group_rows if row[col]!='--']
            if values:
                optimum=(min if direction=='min' else max)(values)
                for row in group_rows:
                    if row[col]!='--' and float(row[col])==optimum:
                        row[col]=r'\textbf{'+row[col]+'}'
    align=align or ('l'+'r'*(len(headers)-1))
    env='longtable' if long else 'tabular'
    lines=[r'\begin{'+env+'}{@{}'+align+r'@{}}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    if long:lines += [r'\endhead']
    lines += [' & '.join(map(str,r))+r' \\' for r in rows]
    lines += [r'\bottomrule',r'\end{'+env+'}']
    (TAB/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'),dpi=220);plt.close(fig)


def summaries():
    main=read('policy_summary.csv')
    discovery=read('dl2019_discovery_summary.csv')
    index={(r['dataset'],r['method'],int(float(r['budget']))):r for r in main}
    raw=read('raw_assessor_metrics.csv')
    labels={'dl2019':'DL 2019','dl2020':'DL 2020','covid_reserved':'COVID (40)','covid_all':'COVID (50)'}
    tex('raw_metrics.tex',['Pool','$N$','Acc.','F1',r'$\kappa$','Brier','ECE','AURC'],[
        [labels[r['dataset']],f"{int(float(r['pairs'])):,}",fmt(val(r,'accuracy'),True,2),fmt(val(r,'macro_f1'),digits=4),fmt(val(r,'quadratic_weighted_kappa_binary'),digits=4),fmt(val(r,'brier_score'),digits=4),fmt(val(r,'ece_probability_bins_10'),digits=4),fmt(val(r,'confidence_aurc_tie_averaged'),digits=4)] for r in raw])
    rows=[]
    for method in METHODS:
        d=next((r for r in discovery if r['method']==('local_unguarded' if method=='selected_local' else method) and r['budget']=='4630'),None)
        a=index['dl2020',method,5693];b=index['covid_reserved',method,2000]
        rows.append([r'\textbf{SAFE-Judge}' if method=='selected_local' else NAMES[method],fmt(val(d,'query_mean_auto_error_rate_mean'),True) if d else '--',fmt(val(a,'query_mean_auto_error_rate'),True),fmt(val(b,'query_mean_auto_error_rate'),True),fmt(val(a,'selective_risk'),True),fmt(val(b,'selective_risk'),True)])
    tex('main_results.tex',['Method','DL19 $E_Q$','DL20 $E_Q$','COVID $E_Q$','DL20 $R$','COVID $R$'],rows,best={i:'min' for i in range(1,6)})
    half=[r for r in main if int(float(r['budget']))=={'dl2020':5693,'covid_reserved':2000,'covid_all':2500}[r['dataset']]]
    tex('half_all_metrics.tex',['Pool / method','$E_Q$','$E_P$','$R$','F1',r'$\kappa$','$C_+$',r'$R_{\max}$','Defined'],[
        [labels[r['dataset']]+' / '+NAMES[r['method']],fmt(val(r,'query_mean_auto_error_rate'),True),fmt(val(r,'auto_error_rate'),True),fmt(val(r,'selective_risk'),True),fmt(val(r,'mixed_macro_f1'),digits=4),fmt(val(r,'mixed_kappa'),digits=4),fmt(val(r,'raw_class_1_coverage'),True,2),fmt(val(r,'worst_class_risk'),True,2),str(int(val(r,'worst_class_risk_defined')))+ '/10' if val(r,'worst_class_risk_defined') is not None else '--'] for r in half],long=True,best={1:'min',2:'min',3:'min',4:'max',5:'max'},groups=[r['dataset'] for r in half])
    ranked=[r for r in half if r['dataset']=='dl2020']
    from scipy.stats import rankdata
    from src.utils.io import write_csv
    system_rows=read('official_dl2020_system_scores.csv');drops=[]
    for method in METHODS:
        for seed in range(6000,6010):
            rr=sorted([r for r in system_rows if r['method']==method and int(r['seed'])==seed],key=lambda r:r['system'])
            assert len(rr)==59
            gold_rank=rankdata(-np.array([float(r['gold_binary_ndcg10']) for r in rr]),method='average')
            mixed_rank=rankdata(-np.array([float(r['mixed_binary_ndcg10']) for r in rr]),method='average')
            drops.append(dict(method=method,seed=seed,maximum_rank_drop=float(np.maximum(mixed_rank-gold_rank,0).max()),
                definition='Largest worsening of system rank within a run; ranks from binary nDCG@10.'))
    write_csv(DATA/'dl2020_maximum_rank_drop.csv',drops)
    drop_mean={m:mean([r for r in drops if r['method']==m],'maximum_rank_drop') for m in METHODS}
    order={m:i for i,m in enumerate(METHODS)};ranked=sorted(ranked,key=lambda r:order[r['method']])
    tex('ranking.tex',['Method',r'$\tau_b$',r'$\rho$','nDCG MAE','Max drop'],[[NAMES[r['method']],fmt(val(r,'system_ranking_kendall_tau'),digits=4),fmt(val(r,'system_ranking_spearman_rho'),digits=4),fmt(val(r,'system_binary_ndcg10_mae'),digits=5),fmt(drop_mean[r['method']],digits=1)] for r in ranked],best={1:'max',2:'max',3:'min',4:'min'})
    tex('ranking_main.tex',['Method',r'$\tau_b$','MAE','Drop'],[[NAMES[r['method']],fmt(val(r,'system_ranking_kendall_tau'),digits=4),fmt(val(r,'system_binary_ndcg10_mae'),digits=4),fmt(drop_mean[r['method']],digits=1)] for r in ranked],best={1:'max',2:'min',3:'min'})
    # Metadata uses the evaluator's existing field names.
    rankkeys=[k for k in ranked[0] if 'system_' in k]
    write_json(REPORT/'ranking_fields.json',rankkeys)
    differences=read('paired_query_intervals.csv')
    tex('uncertainty.tex',['Pool','Comparator',r'$\Delta E_Q$ (pp)',r'95\% interval (pp)'],[
        [labels[r['dataset']],NAMES[r['comparator']],fmt(val(r,'difference'),True),f"[{fmt(val(r,'low'),True)}, {fmt(val(r,'high'),True)}]"] for r in differences if r['comparator'] in ('lara_1_raw','lara_1_upstream_smooth','raw_confidence')])
    tex('budget_sweep.tex',['Pool / method',r'Human \%',r'Coverage \%',r'$E_Q$ \%',r'$E_P$ \%',r'$R$ \%'],[
        [labels[r['dataset']]+' / '+NAMES[r['method']],fmt(val(r,'human_fraction'),True,0),fmt(val(r,'coverage'),True,2),fmt(val(r,'query_mean_auto_error_rate'),True),fmt(val(r,'auto_error_rate'),True),fmt(val(r,'selective_risk'),True)] for r in sorted(main,key=lambda r:(r['dataset'],r['method'],val(r,'human_fraction'))) if r['dataset']!='covid_all' and r['method'] in ('selected_local','lara_1_raw','raw_confidence')],long=True)
    for metric,name,ylabel in [('query_mean_auto_error_rate','budget_error','Query-average auto-error (%)'),('selective_risk','selective_risk','Error among automatic labels (%)')]:
        fig,axes=plt.subplots(1,2,figsize=(6.9,2.45),layout='constrained')
        for ax,dataset,title in zip(axes,['dl2020','covid_reserved'],['DL 2020: 54 topics','TREC-COVID: 40 reserved topics']):
            for method,color in zip(METHODS,COLORS):
                rr=sorted([r for r in main if r['dataset']==dataset and r['method']==method],key=lambda r:val(r,'human_fraction'))
                ax.plot([100*val(r,'human_fraction') for r in rr],[100*val(r,metric) for r in rr],marker='o',ms=2.6,lw=1.8 if method=='selected_local' else 1.0,color=color,label=NAMES[method],ls='--' if method in ('random_review','raw_confidence') else '-')
            ax.set(title=title,xlabel='Human review budget (% of pool)');ax.grid(alpha=.2)
        axes[0].set_ylabel(ylabel);handles,labels_=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels_,loc='outside lower center',ncol=4,frameon=False,fontsize=7.6)
        save(fig,name)
    # Explicitly display both class risks and coverage; empty classes stay missing.
    fig,axes=plt.subplots(1,2,figsize=(7,2.65),layout='constrained')
    for ax,dataset in zip(axes,['dl2020','covid_reserved']):
        rr=[index[dataset,m,5693 if dataset=='dl2020' else 2000] for m in ('selected_local','lara_1_raw','lara_1_upstream_smooth','raw_confidence')]
        x=np.arange(len(rr));w=.35
        ax.bar(x-w/2,[100*val(r,'raw_class_0_coverage') for r in rr],w,label='Raw negative predictions',color='#0072B2')
        ax.bar(x+w/2,[100*val(r,'raw_class_1_coverage') for r in rr],w,label='Raw positive predictions',color='#D55E00')
        ax.set_xticks(x,[NAMES[r['method']] for r in rr],rotation=20,ha='right');ax.set_title(labels[dataset]);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Original-class automatic coverage (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    save(fig,'class_coverage')


def diagnosis():
    q=read('query_error_diagnosis.csv');concentration=read('query_error_concentration.csv')
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.2),layout='constrained')
    for ax,ds,title in zip(axes,['dl2019','dl2020'],['DL 2019 (discovery)','DL 2020 (descriptive check)']):
        rr=sorted([r for r in q if r['dataset']==ds],key=lambda r:val(r,'error_rate'),reverse=True)
        x=np.arange(1,len(rr)+1);k=int(np.ceil(.2*len(rr)))
        ax.bar(x,[100*val(r,'error_rate') for r in rr],color=['#D55E00' if i<k else '#0072B2' for i in range(len(rr))],width=.88)
        ax.set(title=title,xlabel='Queries sorted by observed error rate');ax.grid(axis='y',alpha=.2)
        c=next(r for r in concentration if r['dataset']==ds)
        ax.text(.98,.96,f"Top {k} queries: {100*val(c,'error_share'):.1f}% of errors\n{100*val(c,'pair_share'):.1f}% of pairs",transform=ax.transAxes,ha='right',va='top',fontsize=8)
    axes[0].set_ylabel('Raw assessor error per query (%)');save(fig,'query_errors')
    pred={ds:read(ds+'_predictions.csv') for ds in ('dl2019','dl2020','covid_all')}
    fig,axes=plt.subplots(1,3,figsize=(7,2.3),layout='constrained')
    for ax,(ds,rr),title in zip(axes,pred.items(),['DL 2019','DL 2020','TREC-COVID']):
        y=np.array([int(r['true_binary']) for r in rr]);p=np.array([float(r['prob_relevant'])>=.5 for r in rr]);counts=np.array([[(~p & (y==0)).sum(),(p & (y==0)).sum()],[(~p & (y==1)).sum(),(p & (y==1)).sum()]])
        ax.imshow(counts/counts.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{counts[i,j]:,}\n{100*counts[i,j]/counts[i].sum():.1f}%',ha='center',va='center',color='white' if counts[i,j]/counts[i].sum()>.6 else 'black',fontsize=8)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No','Yes'],yticklabels=['No','Yes'],title=title,xlabel='Assessor prediction')
    axes[0].set_ylabel('Human binary label');save(fig,'confusion')
    bins=read('predicted_class_reliability.csv')
    fig,axes=plt.subplots(1,3,figsize=(7,2.4),layout='constrained')
    for ax,ds,title in zip(axes,['dl2019','dl2020','covid_all'],['DL 2019','DL 2020','TREC-COVID']):
        ax.plot([.5,1],[.5,1],':',color='gray',label='Ideal')
        for c,color in [('0','#0072B2'),('1','#D55E00')]:
            rr=[r for r in bins if r['dataset']==ds and r['predicted_class']==c and val(r,'pairs')>0]
            ax.plot([val(r,'mean_confidence') for r in rr],[val(r,'observed_correctness') for r in rr],'o-',ms=3,color=color,label='Predicted '+('No' if c=='0' else 'Yes'))
        ax.set(title=title,xlabel='Mean confidence',xlim=(.48,1.02),ylim=(0,1.02));ax.grid(alpha=.2)
    axes[0].set_ylabel('Observed correctness');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False);save(fig,'class_calibration')
    # Held-out margins are descriptive. Human grades are never an input to acquisition.
    rr=pred['dl2020'];p=np.clip([float(r['prob_relevant']) for r in rr],1e-7,1-1e-7);margin=np.log(p/(1-p))
    fig,axes=plt.subplots(1,2,figsize=(7,2.5),layout='constrained')
    for g,color in zip(range(4),COLORS[:4]):
        ix=np.array([int(r['true_relevance'])==g for r in rr]);axes[0].hist(margin[ix],bins=np.linspace(-8,8,35),density=True,histtype='step',lw=1.5,color=color,label=f'Grade {g} (n={ix.sum()})')
    axes[0].set(xlabel='Yes-minus-No logit',ylabel='Density',title='DL 2020: overlapping grade scores');axes[0].legend(fontsize=7)
    for c,color in [(0,'#0072B2'),(1,'#D55E00')]:
        conf=np.maximum(p,1-p);y=np.array([int(r['true_binary']) for r in rr]);ix=np.flatnonzero((p>=.5)==c);ix=ix[np.argsort(-conf[ix],kind='stable')];risk=np.cumsum((p[ix]>=.5)!=y[ix])/np.arange(1,len(ix)+1)
        axes[1].plot(100*np.arange(1,len(ix)+1)/len(ix),100*risk,color=color,label='Predicted '+('No' if c==0 else 'Yes'))
    axes[1].axvline(5,ls=':',color='gray',label='5% class floor');axes[1].set(xlabel='Within-class confidence-prefix coverage (%)',ylabel='Accepted-label error (%)',title='DL 2020: confident positive errors');axes[1].legend(fontsize=7);axes[1].grid(alpha=.2)
    save(fig,'grade_and_tail_diagnosis')


def extensions():
    d=read('dl2020_threshold_summary.csv');new=read('supplement_summary.csv')
    threshold_names={**NAMES,'lara_reserved_raw':r'LARA G/raw, 5\% floor',
        'threshold_common_paid':'Confidence, paid calibration',
        'common_paid':'Confidence, paid calibration',
        'threshold_worst_class':'Separate class cutoffs','worst_class':'Separate class cutoffs',
        'total_error':'Cutoffs, total-error objective'}
    rows=[]
    for r in d:
        if r['fraction']=='0.5' and r['calibration_labels'] in ('0','200') and r['method'] in ('threshold_worst_class','threshold_common_paid','raw_confidence','lara_1_raw','lara_1_upstream_smooth','lara_reserved_raw'):
            rows.append(['DL 2020',threshold_names[r['method']],fmt(val(r,'worst_class_risk_mean'),True,2),fmt(val(r,'raw_class_1_coverage_mean'),True,2),fmt(val(r,'coverage_mean'),True,2),r['both_classes_defined']+'/'+r['repetitions']])
    for r in new:
        if r['study']=='separate_thresholds' and r['fraction']=='0.5' and r['calibration_labels']=='200.0':
            rows.append(['COVID (40)',threshold_names[r['method']],fmt(val(r,'worst_class_risk'),True,2),fmt(val(r,'raw_class_1_coverage'),True,2),fmt(val(r,'coverage'),True,2),'20/20'])
    tex('thresholds.tex',['Pool','Rule',r'$R_{\max}$ \%',r'$C_+$ \%',r'Coverage \%','Defined'],rows,align='llrrrr',best={2:'min'},groups=[row[0] for row in rows])
    rows=[]
    for r in new:
        if r['study']=='target_calibration':
            rows.append([str(int(float(r['calibration_labels']))),NAMES[r['method']],{'raw':'Original','platt':'Platt scaled'}[r['score_view']],fmt(val(r,'auto_error_rate'),True),fmt(val(r,'selective_risk'),True),fmt(val(r,'raw_class_1_coverage'),True,2)])
    tex('target_calibration.tex',['Cal. labels','Review rule','Scores',r'$E_P$ \%',r'$R$ \%',r'$C_+$ \%'],rows,align='rllrrr',best={3:'min',4:'min'},groups=[row[0] for row in rows])
    density=read('dl2020_density_prediction.csv')
    grade_names={'raw_assessor':'Original assessor','grade_lda_margin':'Grade LDA, score',
        'grade_qda_margin':'Grade QDA, score','grade_qda_full':'Grade QDA, full features',
        'grade_logistic_full':'Grade logistic, full features','platt_binary':'Binary Platt scaling'}
    tex('density_prediction.tex',['Model','Brier','Error AUC','Grade F1',r'Grade $\kappa_w$'],[[grade_names[r['model']],fmt(val(r,'brier_score_mean'),digits=4),fmt(val(r,'raw_error_auc_mean'),digits=4),fmt(val(r,'grade_macro_f1_mean'),digits=4),fmt(val(r,'grade_quadratic_kappa_mean'),digits=4)] for r in density if r['model'] in grade_names],align='lrrrr',best={1:'min',2:'max',3:'max',4:'max'})
    # Common 33-topic pool: legacy curves share exact IDs; local was recomputed.
    baseline=[]
    for name,method in [('dl2019_lexical_baseline.csv','Lexical active learning'),('dl2019_cross_encoder_baseline.csv','MiniLM + temperature'),('dl2019_lara_baseline.csv','LARA G/norm')]:
        rr=[r for r in read(name) if r['budget']=='3664' and (name!='dl2019_lara_baseline.csv' or r['grouping']=='1')]
        baseline.append([method,fmt(mean(rr,'auto_error_rate'),True),fmt(mean(rr,'selective_risk'),True),fmt(mean(rr,'coverage'),True,2),str(int(mean(rr,'calibration_labels') or 0))])
    rr=[r for r in new if r['study']=='common_dl2019']
    baseline.append(['Local',fmt(mean(rr,'auto_error_rate'),True),fmt(mean(rr,'selective_risk'),True),fmt(mean(rr,'coverage'),True,2),'0'])
    quality={r['method']:r for r in read('common_baseline_metrics.csv')}
    baseline=[r+[fmt(val(quality[r[0]],'mixed_macro_f1'),digits=4),fmt(val(quality[r[0]],'mixed_kappa'),digits=4)] for r in baseline]
    main_labels={'Lexical active learning':'Lexical AL','MiniLM + temperature':'MiniLM + T','Local':'SAFE-Judge','LARA G/norm':'LARA G/norm'}
    tex('classical_neural_main.tex',['Method',r'$E_P$ \%',r'$R$ \%',r'$\gamma$ \%','F1'],[[main_labels[r[0]],r[1],r[2],r[3],r[5]] for r in baseline],best={1:'min',2:'min',4:'max'})
    baseline=[['SAFE-Judge' if r[0]=='Local' else r[0]]+r[1:] for r in baseline]
    tex('classical_neural.tex',['Method',r'$E_P$ \%',r'$R$ \%',r'Coverage \%','Cal. labels','F1',r'$\kappa$'],baseline,align='lrrrrrr',best={1:'min',2:'min',5:'max',6:'max'})
    ab=read('dl2019_mechanism_summary.csv')
    rr=[r for r in ab if r['budget']=='4630' and r['method'] in ('real_local_pair','real_shared_pair','shuffled_local_pair','shuffled_shared_pair')]
    mechanism_names={'real_local_pair':'SAFE-Judge / real queries',
        'real_shared_pair':'Shared-only / real queries',
        'shuffled_local_pair':'SAFE-Judge / shuffled groups',
        'shuffled_shared_pair':'Shared-only / shuffled groups'}
    tex('mechanism.tex',['Grouping / update',r'$E_Q$ \%'],[[mechanism_names[r['method']],fmt(val(r,'query_mean_auto_error_rate_mean'),True)] for r in rr],best={1:'min'})
    a=read('dl2020_class_aware_summary.csv')
    rr=[r for r in a if r['fraction']=='0.5' and r['method'] in ('local_minimax_floor05','local_minimax_floor20','shared_minimax_floor05','local_query_error_floor05','selected_local')]
    class_names={'selected_local':'SAFE-Judge','local_minimax_floor05':r'Minimax, query feedback, 5\% floor',
        'local_minimax_floor20':r'Minimax, query feedback, 20\% floor',
        'shared_minimax_floor05':r'Minimax, shared-only, 5\% floor',
        'local_query_error_floor05':r'Query-error objective, 5\% floor'}
    tex('class_aware.tex',['Allocation',r'$E_Q$ \%',r'$R_{\max}$ \%',r'$C_+$ \%','Defined'],[[class_names[r['method']],fmt(val(r,'query_mean_auto_error_rate'),True),fmt(val(r,'worst_class_risk'),True,2),fmt(val(r,'raw_class_1_coverage'),True,2),r['worst_class_risk_defined']+'/10'] for r in rr],best={1:'min',2:'min'},groups=['floor05' if r['method'] in ('local_minimax_floor05','shared_minimax_floor05','local_query_error_floor05') else None for r in rr])
    efficiency=read('inference_efficiency.csv')
    tex('efficiency.tex',['Pool','GPU time (min)','Pairs/s','Truncated'],[[{'dl2019':'DL 2019','dl2020':'DL 2020','covid_all':'COVID'}[r['dataset']],fmt(val(r,'inference_seconds')/60,digits=2),fmt(val(r,'pairs_per_second'),digits=2),str(int(val(r,'truncated_pairs')))] for r in efficiency])
    # Additional calibration sweep, all costs retained.
    rr=[r for r in new if r['study']=='separate_thresholds' and r['method']=='worst_class']
    tex('covid_threshold_sweep.tex',['Cal. labels',r'Total human \%',r'$R_{\max}$ \%',r'$E_Q$ \%',r'Coverage \%'],[[str(int(float(r['calibration_labels']))),fmt(val(r,'fraction'),True,0),fmt(val(r,'worst_class_risk'),True,2),fmt(val(r,'query_mean_auto_error_rate'),True),fmt(val(r,'coverage'),True,2)] for r in rr])
    fig,axes=plt.subplots(1,2,figsize=(7,2.5),layout='constrained')
    for cost,color in zip((100,200,500),COLORS[:3]):
        rr=sorted([r for r in new if r['study']=='separate_thresholds' and r['method']=='worst_class' and int(float(r['calibration_labels']))==cost],key=lambda r:val(r,'fraction'))
        axes[1].plot([100*val(r,'fraction') for r in rr],[100*val(r,'worst_class_risk') for r in rr],'o-',ms=3,color=color,label=f'{cost} calibration labels')
        rr=sorted([r for r in d if r['method']=='threshold_worst_class' and r['calibration_labels']==str(cost)],key=lambda r:val(r,'fraction'))
        axes[0].plot([100*val(r,'fraction') for r in rr],[100*val(r,'worst_class_risk_mean') for r in rr],'o-',ms=3,color=color,label=f'{cost} calibration labels')
    for ax,title in zip(axes,['DL 2020','TREC-COVID (40 reserved topics)']):ax.set(title=title,xlabel='Total human-label budget (% of test pool)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Worst accepted-class error (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=7.5);save(fig,'threshold_calibration_cost')


def supplementary_tables():
    rr=read('predicted_class_calibration.csv')
    tex('class_calibration.tex',['Pool','Prediction','Pairs',r'Error \%','Conf. ECE','Brier'],[[r['dataset'].replace('_',' '),'No' if r['predicted_class']=='0' else 'Yes',str(int(val(r,'pairs'))),fmt(val(r,'error_rate'),True,2),fmt(val(r,'confidence_ece_10_bins'),digits=4),fmt(val(r,'brier_score'),digits=4)] for r in rr],align='llrrrr')
    rr=read('covid_threshold_query_intervals.csv')
    tex('covid_threshold_intervals.tex',['Comparator',r'$\Delta R_{\max}$ (pp)',r'95\% interval (pp)'],[[NAMES.get(r['comparator'],r['comparator'].replace('_',' ')),fmt(val(r,'difference'),True,2),f"[{fmt(val(r,'low'),True,2)}, {fmt(val(r,'high'),True,2)}]"] for r in rr])
    rr=read('oosterhuis_trecdl_reference.csv')
    tex('oosterhuis_reference.tex',['Calibration queries','Method','CI coverage','Mean width'],[[r['calibration_queries'],r['method'].upper(),fmt(val(r,'interval_coverage'),digits=2),fmt(val(r,'mean_width'))] for r in rr])
    rr=read('oosterhuis_trecdl_matched.csv')
    matched=[]
    for r in rr:
        values=[val(r,key) for key in ('local_dcg_mae','best_lara_dcg_mae','ppi_dcg_mae','crc_midpoint_dcg_mae')]
        lowest=min(values)
        cells=[r'\textbf{'+fmt(value,digits=4)+'}' if value==lowest else fmt(value,digits=4) for value in values]
        matched.append([r['human_labels']]+cells)
    tex('oosterhuis_matched.tex',['Human labels','SAFE-Judge','Best LARA','PPI','CRC midpoint'],matched)
    rr=read('dl2019_target_pool_crc.csv')
    tex('sparse_crc.tex',['Rule','Queries','Paid pilot',r'Coverage \%',r'$E_Q$ \%','Total human'],[[r['method'],r['calibration_queries'],r['calibration_labels'],fmt(val(r,'coverage_mean'),True,2),fmt(val(r,'query_mean_auto_error_rate_mean'),True,2),fmt(val(r,'total_human_labels_mean'),digits=1)] for r in rr])
    rr=read('dl2019_progressive_audit.csv')
    tex('progressive_audit.tex',['Rule','Filter','Releases','Total human'],[[NAMES.get(r['method'],r['method'].replace('_',' ')),r['feasibility_gate'],r['releases']+'/'+r['runs'],fmt(val(r,'mean_total_labels'),digits=1)] for r in rr],align='llrr')
    rr=json.loads((DATA/'manual_failure_review.metadata.json').read_text())
    tex('manual_taxonomy.tex',['Category','Count'],[[k.replace('_',' ').capitalize(),str(v)] for k,v in rr['category_counts'].items()])
    rr=[r for r in read('policy_summary.csv') if r['method'] in ('selected_local','lara_1_raw','raw_confidence') and r['human_fraction']=='0.5']
    tex('query_tails.tex',['Pool / rule',r'Mean $E_Q$ \%',r'Tail $E_Q$ \%',r'Worst query \%'],[[r['dataset'].replace('_',' ')+' / '+NAMES[r['method']],fmt(val(r,'query_mean_auto_error_rate'),True),fmt(val(r,'tail20_query_auto_error'),True),fmt(val(r,'worst_query_auto_error'),True)] for r in rr])
    rr=[r for r in read('dl2020_density_review.csv') if r['accounting']=='matched_total' and r['nominal_fraction']=='0.5']
    # The input's accounting values are retained exactly; select the total-cost view.
    if not rr:rr=[r for r in read('dl2020_density_review.csv') if 'total' in r['accounting'] and r['nominal_fraction']=='0.5']
    review_names={'grade_qda_margin_risk_only':'Grade QDA, score',
        'grade_lda_margin_risk_only':'Grade LDA, score',
        'binary_lda_margin_risk_only':'Binary LDA, score',
        'grade_logistic_full_risk_only':'Grade logistic, full features',
        'platt_binary_risk_only':'Binary Platt scaling'}
    tex('density_review.tex',['Rule',r'$E_Q$ \%',r'$R$ \%',r'Coverage \%'],[[review_names[r['method']],fmt(val(r,'query_mean_auto_error_rate_mean'),True),fmt(val(r,'selective_risk_mean'),True),fmt(val(r,'coverage_mean'),True,2)] for r in rr if r['method'] in review_names],align='lrrr',best={1:'min',2:'min'})
    # Official rank changes calculated from average system scores, not label error.
    from scipy.stats import rankdata
    rr=read('official_dl2020_system_scores.csv');systems=sorted({r['system'] for r in rr})
    gold=np.array([next(float(r['gold_binary_ndcg10']) for r in rr if r['system']==s) for s in systems])
    goldrank=rankdata(-gold)
    fig,axes=plt.subplots(1,2,figsize=(7,2.7),layout='constrained')
    changes=[]
    for ax,method in zip(axes,['selected_local','lara_1_raw']):
        scores=np.array([np.mean([float(r['mixed_binary_ndcg10']) for r in rr if r['system']==s and r['method']==method]) for s in systems]);ranks=rankdata(-scores)
        ax.scatter(gold,scores,s=13,color=COLORS[0 if method=='selected_local' else 1],alpha=.8)
        lo=min(gold.min(),scores.min())-.01;hi=max(gold.max(),scores.max())+.01
        ax.plot([lo,hi],[lo,hi],':',color='gray');ax.set(title=NAMES[method],xlabel='Human-reference binary nDCG@10',xlim=(lo,hi),ylim=(lo,hi));ax.grid(alpha=.2)
        for s,g,p,a,b in zip(systems,gold,scores,goldrank,ranks):changes.append(dict(system=s,method=method,gold_score=float(g),mixed_score=float(p),gold_rank=float(a),mixed_rank=float(b),rank_change=float(b-a)))
    axes[0].set_ylabel('Mixed-label binary nDCG@10');save(fig,'ranking_fidelity')
    from src.utils.io import write_csv
    write_csv(DATA/'official_dl2020_mean_rank_changes.csv',changes)
    # Sampled adaptive-policy area remains distinct from exact scorer AURC.
    rr=read('policy_summary.csv');area=[]
    for ds in ('dl2020','covid_reserved','covid_all'):
        for method in METHODS:
            x=sorted([r for r in rr if r['dataset']==ds and r['method']==method],key=lambda r:val(r,'coverage'))
            cover=np.array([val(r,'coverage') for r in x]);risk=np.array([val(r,'selective_risk') for r in x]);v=float(np.trapezoid(risk,cover))
            area.append(dict(dataset=ds,method=method,coverage_lower=float(cover.min()),coverage_upper=float(cover.max()),sampled_partial_aurc=v,range_normalized_area=v/(cover.max()-cover.min())))
    write_csv(DATA/'adaptive_partial_aurc.csv',area)


def main():
    FIG.mkdir(parents=True,exist_ok=True);TAB.mkdir(parents=True,exist_ok=True)
    summaries();diagnosis();extensions();supplementary_tables()
    write_json(REPORT/'build_validation.json',dict(builder_sha256=file_hash(Path(__file__)),input_sha256={p.name:file_hash(p) for p in sorted(DATA.glob('*')) if p.is_file() and p.name!='SHA256SUMS.txt'},
        figures=[p.name for p in sorted(FIG.glob('*.pdf'))],tables=[p.name for p in sorted(TAB.glob('*.tex'))],status='generated; compile and visual QA recorded separately'))
    print('Final tables and figures built',flush=True)


if __name__=='__main__':main()
