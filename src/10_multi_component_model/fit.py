import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '10_multi_component_model'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,pickle,json,warnings,pandas as pd;warnings.filterwarnings('ignore')
C=np.load('comps.npz');CL=list(C['CL']);iu=np.triu_indices(5,1)
COMP=['vision','touch','geometry','word']
M=np.column_stack([(lambda v:(v-v.mean())/v.std())(C[f'rdm_{k}'][iu]) for k in COMP])
G=dict(zip(*pd.read_excel('../../participants/participant_information.xlsx')[['code','analysis_group']].values.T))
D=pickle.load(open('../../data/derived/06_features/feats.pkl','rb'))
CONDS={'B1_erp':'Block 1 · word onset','B1_img':'Block 1 · imagery','B2_touch':'Block 2 · touch','B2_img':'Block 2 · imagery','B3_erp':'Block 3 · picture onset','B3_perc':'Block 3 · picture viewing','B3_img':'Block 3 · imagery'}
def z(X):return (X-X.mean(0))/(X.std(0)+1e-9)
def seqs(c):
    S={n:dict(zip(D[n][c]['pos'].tolist(),D[n][c]['y'].tolist())) for n in D if c in D[n]};com={}
    for n,m in S.items():
        if not(n=='P23' and c.startswith('B2')):[com.setdefault(p,l) for p,l in m.items()]
    for n in S:
        b=dict(com) if not(n=='P23' and c.startswith('B2')) else {15:'pyramid'};b.update(S[n]);S[n]=b
    return S
def nrdm(X,y):
    A=[];B=[]
    for k in CL:
        i=np.where(y==k)[0]
        if len(i)<2: return None
        A.append(X[i[0::2]].mean(0));B.append(X[i[1::2]].mean(0))
    Cc=np.corrcoef(np.array(A),np.array(B))[:5,5:];Dm=1-(Cc+Cc.T)/2;v=Dm[iu];return (v-v.mean())/(v.std()+1e-9)
def betas(c,perm=None,S=None):
    out={}
    for n in D:
        if c not in D[n]: continue
        y=D[n][c]['y'] if perm is None else np.array([S[n][perm[p-1]] for p in D[n][c]['pos']])
        v=nrdm(z(D[n][c]['X']),y)
        if v is None: continue
        out[n]=np.linalg.lstsq(np.c_[np.ones(10),M],v,rcond=None)[0][1:]
    return out
rng=np.random.default_rng(42);P=[rng.permutation(np.arange(1,16)) for _ in range(1000)];res={}
for c,nm in CONDS.items():
    S=seqs(c);b=betas(c);Bm=np.mean(list(b.values()),0)
    nl=np.array([np.mean(list(betas(c,p,S).values()),0) for p in P]);p=(np.sum(nl>=Bm,0)+1)/1001
    grp={}
    for gname in ['Sighted','Congenitally low vision','Later vision loss','Congenitally blind']:
        ks=[k for k in b if G[k]==gname]
        if ks: grp[gname]=dict(n=len(ks),beta=np.mean([b[k] for k in ks],0).round(3).tolist())
    res[c]=dict(ad=nm,n=len(b),beta=Bm.tolist(),p=p.tolist(),null95=np.percentile(nl,95,0).tolist(),group=grp,person={k:v.tolist() for k,v in b.items()})
    print(f'{nm:24s} n={len(b):2d} | '+' | '.join(f'{k} β={x:+.3f} (p={pp:.3f})' for k,x,pp in zip(COMP,Bm,p)),flush=True)
json.dump(dict(component=COMP,result=res),open('fit_res.json','w'),ensure_ascii=False)
