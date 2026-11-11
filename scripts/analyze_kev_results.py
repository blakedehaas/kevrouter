#!/usr/bin/env python3
"""Reproduce development-selected Kev policy, fixed baseline and paired analyses."""
import csv, json, os, sys
from pathlib import Path
from collections import defaultdict
os.environ.setdefault('MPLCONFIGDIR','/tmp/kevrouter-matplotlib')
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(Path(__file__).resolve().parent.parent/'data/.cache/tiktoken'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tiktoken
from run_kev_experiment_v3 import COSTS

ROOT=Path(__file__).resolve().parent.parent
RUN=ROOT/'outputs/experiments/kev-4b-9b-v3-g13qWFcv'
OLD=ROOT/'outputs/experiments/kev-4b-9b-a100-P6C3DFvE'
OUT=ROOT/'outputs/analysis/kev-routing-2026-11-10'
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261110
BUDGETS=[.001,.002,.005,.01,.02]
MUS=np.unique(np.r_[0.,np.geomspace(0.0001,100,501),2.5])
ENC=tiktoken.get_encoding('cl100k_base')
def read(path):
    with open(path) as f: return [json.loads(x) for x in f if x.strip()]
def csvwrite(name, rows):
    if not rows:return
    with open(OUT/name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def load_split(split):
    raw=read(ROOT/'data/processed'/f"{'val' if split=='development' else split}.jsonl")
    files={s:read(RUN/f'finetuned/kev-{s}-finetuned/{split}_predictions.jsonl') for s in ['4b','9b']}
    n=len(files['4b']); assert all(len(x)==n for x in files.values())
    sources=[]
    for i,p in enumerate(files['4b']):
        r=raw[p['processed_row']]
        assert (r['dataset'],r['index'])==(p['dataset'],p['index'])
        for s in files:
            q=files[s][i]; assert (q['dataset'],q['index'],q['processed_row'])==(p['dataset'],p['index'],p['processed_row'])
            assert set(q['eligible_probabilities'])==set(r['model_evals'])
        sources.append(r)
    assert len({r['query_hash'] for r in sources})<=n
    return sources,files

def make_arrays(raw,files):
    models=sorted(set.intersection(*(set(r['model_evals']) for r in raw)))
    assert all(set(r['model_evals'])==set(models) for r in raw)
    scores=np.array([[float(r['model_evals'][m]['score'])>=1 for m in models] for r in raw],dtype=np.float64)
    costs=np.array([[float(r['model_evals'][m]['cost']) for m in models] for r in raw])
    price=np.array([COSTS[m] for m in models]); probs={s:np.array([[p['eligible_probabilities'][m] for m in models] for p in files[s]]) for s in files}
    assert all(np.allclose(v.sum(axis=1),1) for v in probs.values())
    return models,scores,costs,price,probs

def eval_choice(choice,sc,co):
    i=np.arange(len(choice)); return sc[i,choice],co[i,choice]
def val_choice(probs,price,mu):return np.argmax(probs-mu*price,axis=1)
def metrics(choice,sc,co):
    a,c=eval_choice(choice,sc,co);return float(a.mean()*100),float(c.mean())
def official_choice(rows,models,pol):return np.array([models.index(r['policies'][pol]['chosen']) for r in rows])

data={k:load_split(k) for k in ['development','test_id','test_ood']}
sets={s:{r['query_hash'] for r in data[s][0]} for s in data}
assert not sets['development'] & sets['test_id'] and not sets['development'] & sets['test_ood'] and not sets['test_id'] & sets['test_ood']
arr={k:make_arrays(*v) for k,v in data.items()}
# Confirm our reconstructed scalarized policy exactly reproduces saved decisions.
for split,(raw,files) in data.items():
    models,sc,co,price,pr=arr[split]
    for s in pr:
        assert np.array_equal(val_choice(pr[s],price,2.5),official_choice(files[s],models,'cost_utility_mu_2.5'))
        assert np.array_equal(val_choice(pr[s],price,0),official_choice(files[s],models,'argmax'))

# Selection is made once from development, by maximum accuracy under each fixed budget;
# ties choose cheaper then smaller mu. Fixed per-model costs are the decision-time proxies.
selected={}; sweep=[]
models,sc,co,price,pr=arr['development']
for s in ['4b','9b']:
    candidates=[]
    for mu in MUS:
        ac,ct=metrics(val_choice(pr[s],price,mu),sc,co)
        candidates.append((float(mu),ac,ct)); sweep.append(dict(model=s,mu=float(mu),accuracy_pct=ac,cost_per_query=ct))
    for budget in BUDGETS:
        eligible=[x for x in candidates if x[2]<=budget]
        selected[(s,budget)]=min(eligible,key=lambda x:(-x[1],x[2],x[0])) if eligible else None
csvwrite('development_sweep.csv',sweep)
selrows=[]; evalrows=[]; fixedrows=[]; policies={}
for split,(raw,files) in data.items():
    models,sc,co,price,pr=arr[split];n=len(raw); policies[split]={}
    for j,m in enumerate(models):
        ch=np.full(n,j);a,c=metrics(ch,sc,co); fixedrows.append(dict(split=split,model=m,n=n,accuracy_pct=a,cost_per_query=c))
    for s in ['4b','9b']:
        for pol in ['argmax','cost_utility_mu_2.5','paper_tier_gate_tau_0.6']:
            ch=official_choice(files[s],models,pol);policies[split][f'{s}_3epoch_{pol}']=ch
            a,c=metrics(ch,sc,co);evalrows.append(dict(split=split,model=s,policy=pol,mu=0 if pol=='argmax' else (2.5 if pol=='cost_utility_mu_2.5' else ''),budget='',n=n,accuracy_pct=a,cost_per_query=c))
        for budget in BUDGETS:
            v=selected[(s,budget)]
            if v is None: continue
            mu,devacc,devcost=v;ch=val_choice(pr[s],price,mu);policies[split][f'{s}_budget_{budget}']=ch
            a,c=metrics(ch,sc,co);evalrows.append(dict(split=split,model=s,policy='dev_selected',mu=mu,budget=budget,n=n,accuracy_pct=a,cost_per_query=c))
            if split=='development': selrows.append(dict(model=s,budget=budget,mu=mu,development_accuracy_pct=devacc,development_cost=devcost))
    # The original run only stored argmax predictions.
    for s in ['4b','9b']:
        old=read(OLD/f'kev-{s}-finetuned/{split}_predictions.jsonl')
        assert len(old)==n
        assert all((r['dataset'],r['index'],r['processed_row'])==(q['dataset'],q['index'],q['processed_row']) for r,q in zip(old,files[s]))
        policies[split][f'{s}_1epoch_argmax']=np.array([models.index(r['chosen']) for r in old])
        ch=policies[split][f'{s}_1epoch_argmax'];a,c=metrics(ch,sc,co)
        assert all(bool(sc[i,ch[i]])==bool(r['correct']) and np.isclose(co[i,ch[i]],r['cost']) for i,r in enumerate(old))
        evalrows.append(dict(split=split,model=s,policy='1epoch_argmax',mu='',budget='',n=n,accuracy_pct=a,cost_per_query=c))
    policies[split]['fixed_qwen']=np.full(n,models.index('qwen3-235b-a22b-2507'))
csvwrite('development_selected.csv',selrows);csvwrite('policy_results.csv',evalrows);csvwrite('fixed_baselines.csv',fixedrows)

# Cluster bootstrap by query_hash to account for duplicated query text, preserving paired outcomes.
bootstrap=[];rng=np.random.default_rng(SEED)
comparisons=[('4b_3epoch_argmax','9b_3epoch_argmax'),('4b_1epoch_argmax','9b_1epoch_argmax'),('4b_3epoch_argmax','4b_1epoch_argmax'),('9b_3epoch_argmax','9b_1epoch_argmax'),('4b_3epoch_argmax','fixed_qwen'),('9b_3epoch_argmax','fixed_qwen'),('4b_1epoch_argmax','fixed_qwen'),('9b_1epoch_argmax','fixed_qwen'),('9b_budget_0.005','fixed_qwen'),('4b_budget_0.005','fixed_qwen')]
for split in ['test_id','test_ood']:
    raw,_=data[split];models,sc,co,price,pr=arr[split];hashes=[r['query_hash'] for r in raw];uniq,inv=np.unique(hashes,return_inverse=True);g=len(uniq)
    group_n=np.bincount(inv,minlength=g); W=rng.multinomial(g,np.full(g,1/g),size=5000); den=W@group_n
    for a,b in comparisons:
        ca,cb=policies[split][a],policies[split][b]; xa,za=eval_choice(ca,sc,co);xb,zb=eval_choice(cb,sc,co)
        delta=xa-xb;dc=za-zb; reps=(W@np.bincount(inv,weights=delta,minlength=g))/den*100; creps=(W@np.bincount(inv,weights=dc,minlength=g))/den
        bootstrap.append(dict(split=split,comparison=f'{a} minus {b}',n=len(raw),query_hash_clusters=g,accuracy_a_pct=100*xa.mean(),accuracy_b_pct=100*xb.mean(),delta_pp=100*delta.mean(),ci95_low_pp=np.percentile(reps,2.5),ci95_high_pp=np.percentile(reps,97.5),a_only=int(((xa==1)&(xb==0)).sum()),b_only=int(((xa==0)&(xb==1)).sum()),same_destination=int((ca==cb).sum()),cost_delta=dc.mean(),cost_ci95_low=np.percentile(creps,2.5),cost_ci95_high=np.percentile(creps,97.5)))
csvwrite('paired_bootstrap.csv',bootstrap)

# ID breakdowns. Length quartile cutpoints are learned on development prompts only.
dev_len=np.array([len(ENC.encode(r['prompt'],disallowed_special=())) for r in data['development'][0]])
cuts=np.quantile(dev_len,[.25,.5,.75]);raw,_=data['test_id'];models,sc,co,price,pr=arr['test_id'];length=np.array([len(ENC.encode(r['prompt'],disallowed_special=())) for r in raw]); lbin=np.searchsorted(cuts,length,side='right')
breakdowns=[]
for pname in ['fixed_qwen','4b_1epoch_argmax','9b_1epoch_argmax','4b_3epoch_argmax','9b_3epoch_argmax','4b_3epoch_cost_utility_mu_2.5','9b_3epoch_cost_utility_mu_2.5']:
    ch=policies['test_id'][pname];a,c=eval_choice(ch,sc,co)
    for typ,values in [('benchmark',[r['dataset'] for r in raw]),('prompt_length_quartile',[f'Q{x+1}' for x in lbin]),('selected_model',[models[i] for i in ch]),('decision_cost_tier',[('cheap <= $0.01' if price[i] <= .01 else 'mid $0.01-$0.05' if price[i] <= .05 else 'high > $0.05') for i in ch])]:
        for label in sorted(set(values)):
            mask=np.array([v==label for v in values]);breakdowns.append(dict(policy=pname,dimension=typ,group=label,n=int(mask.sum()),accuracy_pct=float(a[mask].mean()*100),cost_per_query=float(c[mask].mean()),total_cost=float(c[mask].sum())))
csvwrite('id_breakdowns.csv',breakdowns)
# Development full curves; ID/OOD only preselected candidates. All fixed models shown.
for split in ['development','test_id','test_ood']:
    fig,ax=plt.subplots(figsize=(12,8))
    for s,color in [('4b','#0072b2'),('9b','#d55e00')]:
        if split=='development':
            points=[r for r in sweep if r['model']==s];ax.plot([r['cost_per_query'] for r in points],[r['accuracy_pct'] for r in points],color=color,alpha=.5,lw=1,label=f'{s} development sweep')
        points=[r for r in evalrows if r['split']==split and r['model']==s and r['policy']=='dev_selected'];points=sorted(points,key=lambda x:x['mu']);ax.plot([r['cost_per_query'] for r in points],[r['accuracy_pct'] for r in points],marker='o',color=color,lw=1.3,label=f'{s} development-selected')
        for r in points:ax.annotate(f"{r['budget']:.3f}",(r['cost_per_query'],r['accuracy_pct']),fontsize=6,color=color,xytext=(3,3),textcoords='offset points')
    for r in [x for x in fixedrows if x['split']==split]:
        ax.scatter(r['cost_per_query'],r['accuracy_pct'],marker='x',color='black',s=24);ax.annotate(r['model'],(r['cost_per_query'],r['accuracy_pct']),fontsize=6,xytext=(3,3),textcoords='offset points')
    ax.set_xscale('log');ax.set_xlabel('Mean recorded downstream cost per query (USD, log scale)');ax.set_ylabel('Routing accuracy (%)');ax.set_title(f'{split}: fixed models and development-selected policies');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/f'{split}_accuracy_cost.png',dpi=170);plt.close(fig)
meta=dict(seed=SEED,bootstrap_replicates=5000,bootstrap='paired nonparametric bootstrap clustered by query_hash; percentile intervals',mu_grid_size=len(MUS),mu_min=float(MUS.min()),mu_max=float(MUS.max()),budgets=BUDGETS,selection='highest development accuracy subject to mean development recorded downstream cost <= budget; ties lower cost then lower mu; infeasible budgets omitted',decision_costs=COSTS,development_prompt_length_cutpoints_cl100k=cuts.tolist(),notes=['Decision uses fixed per-model proxy costs; evaluation uses per-query recorded costs.', 'Development-selected settings evaluated unchanged on ID and exploratory OOD.', 'Three epochs at 2e-4 compared with one epoch at 2e-5; not a single-variable epoch ablation.', 'All prices and scores are dataset measurements; router execution cost is excluded.'])
(OUT/'analysis_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Output:',OUT)
print('Selected:',selrows)
for x in bootstrap:
 if x['split']=='test_id':print(x['comparison'],round(x['delta_pp'],2),round(x['ci95_low_pp'],2),round(x['ci95_high_pp'],2),'unique',x['a_only'],x['b_only'])
