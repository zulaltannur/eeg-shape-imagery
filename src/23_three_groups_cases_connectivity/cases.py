import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '23_three_groups_cases_connectivity'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Single cases (P10, P23): position relative to the three groups with Crawford–Howell
import numpy as np,json,pickle,warnings,os,sys;warnings.filterwarnings('ignore')
from scipy.stats import t as tdist
exec(open('../../src/23_three_groups_cases_connectivity/matrix.py').read().split('rng=np.random.default_rng(3)')[0])
rng=np.random.default_rng(8);R=feats('B2_img');yy={n:np.array([CL.index(q) for q in R[n]['y']]) for n in R}
grp={g:sorted(n for n in R if G(n)==g) for g in ['CLV','LVL','Sighted']}
def acc(te,pool):
    pool=[x for x in pool if x!=te];P=np.mean([hier(R,R,list(rng.choice(pool,4,replace=False)),te,'X') for _ in range(20)],0);return float(np.mean(P.argmax(1)==yy[te]))
idx={}
for n in R:
    if G(n) is None: continue
    idx[n]=dict(clv_pattern=acc(n,grp['CLV']),sighted_pattern=acc(n,grp['Sighted']));idx[n]['closeness']=idx[n]['clv_pattern']-idx[n]['sighted_pattern']
    print(n,G(n),{k:round(v,3) for k,v in idx[n].items()},flush=True)
z=json.load(open('../../results/23_three_groups_cases_connectivity/connectivity_res.json'))['occipital_second']
def ch(x,ctrl):   # Crawford–Howell: single person against a small control group
    ctrl=np.array(ctrl);n=len(ctrl);t=(x-ctrl.mean())/(ctrl.std(ddof=1)*np.sqrt((n+1)/n));return float(t),float(2*tdist.sf(abs(t),n-1))
out={'person':idx,'case':{}}
for case in ['P10','P23']:
    o={}
    for meas,get in [('pattern closeness (congenitally limited pattern − sighted pattern)',lambda n:idx[n]['closeness']),
                     ('occipital relative alpha · touch',lambda n:np.mean(z['B2_touch'][n])),('occipital relative alpha · imagery',lambda n:np.mean(z['B2_img'][n]))]:
        x=get(case);o[meas]={'value':round(float(x),3)}
        for g in ['CLV','LVL','Sighted']:
            ctrl=[get(n) for n in grp[g] if n!=case and (n in idx)]
            t,p=ch(x,ctrl);o[meas][g]=dict(mean=round(float(np.mean(ctrl)),3),n=len(ctrl),t=round(t,2),p=round(p,3))
    out['case'][case]=o;print(case,json.dumps(o,ensure_ascii=False))
json.dump(out,open('../../results/23_three_groups_cases_connectivity/case_res.json','w'),ensure_ascii=False,indent=1)
