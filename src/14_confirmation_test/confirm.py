import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '14_confirmation_test'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,pickle,json,warnings,pandas as pd;warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
E=pickle.load(open('../../data/derived/07_things/ours_emb.pkl','rb'))
G=dict(zip(*pd.read_excel('../../participants/participant_information.xlsx')[['code','analysis_group']].values.T))
def z(X):return (X-X.mean(0))/(X.std(0)+1e-9)
def seqs(c):
    S={n:dict(zip(E[n][c]['pos'].tolist(),E[n][c]['y'].tolist())) for n in E if c in E[n]};com={}
    for n,m in S.items():
        if not(n=='P23' and c.startswith('B2')):[com.setdefault(p,l) for p,l in m.items()]
    for n in S:
        b=dict(com) if not(n=='P23' and c.startswith('B2')) else {15:'pyramid'};b.update(S[n]);S[n]=b
    return S
def run(a,b,perm=None,excl=(),S=None):
    names=[n for n in E if a in E[n] and b in E[n] and n not in excl];pool=[n for n in E if a in E[n] and n not in excl];acc={}
    for n in names:
        tr=[m for m in pool if m!=n]
        y=np.concatenate([E[m][a]['y'] if perm is None else np.array([S[m][perm[p-1]] for p in E[m][a]['pos']]) for m in tr])
        c=make_pipeline(StandardScaler(),LogisticRegression(C=0.01,max_iter=3000)).fit(np.vstack([z(E[m][a]['Z']) for m in tr]),y)
        acc[n]=float(c.score(z(E[n][b]['Z']),E[n][b]['y']))
    return acc
rng=np.random.default_rng(777);P=[rng.permutation(np.arange(1,16)) for _ in range(1000)];out={}
for a,b in [('B1_img','B1_img'),('B2_touch','B2_img')]:
    S=seqs(a);r=run(a,b);m=np.mean(list(r.values()))
    from joblib import Parallel,delayed;nl=np.array(Parallel(n_jobs=2)(delayed(lambda p:np.mean(list(run(a,b,p,S=S).values())))(p) for p in P));p=(np.sum(nl>=m)+1)/1001
    r2=run(a,b,excl=('P19','P10'));nl2=np.array(Parallel(n_jobs=2)(delayed(lambda p:np.mean(list(run(a,b,p,('P19','P10'),S).values())))(p) for p in P[:300]));p2=(np.sum(nl2>=np.mean(list(r2.values())))+1)/301
    grp={g:round(float(np.mean([v for k,v in r.items() if G[k]==g])),3) for g in set(G[k] for k in r)}
    out[f'{a}->{b}']=dict(acc=float(m),p=float(p),p_bonf=float(min(1,2*p)),null95=float(np.percentile(nl,95)),acc_excl=float(np.mean(list(r2.values()))),p_excl=float(p2),group=grp,person=r)
    print(a,'->',b,f'accuracy {m:.3f} p={p:.4f} Bonferroni p={min(1,2*p):.4f} | P19+P10 excl {np.mean(list(r2.values())):.3f} p={p2:.3f} | groups {grp}',flush=True)
json.dump(out,open('confirm_res.json','w'),ensure_ascii=False)
