import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '18_detailed_analyses'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Stage 18: confusion patterns, within-person similarity across blocks, group comparison
import numpy as np,json,pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from itertools import permutations,combinations
from scipy.stats import kruskal
rng=np.random.default_rng(0);NP=5000
CL=['cone','cube','sphere','pyramid','cylinder']
R=json.load(open('../../results/16_accuracy_boosting/final_res.json'))
W=['Block 1 · word onset','Block 1 · imagery','Block 2 · touch','Block 2 · imagery','Block 2 · touch to imagery','Block 3 · picture onset','Block 3 · picture viewing','Block 3 · imagery']
def soft(P,T):L=np.log(P+1e-9)/T;E=np.exp(L-L.max(1,keepdims=True));return E/E.sum(1,keepdims=True)
cal={}
for w,res in R.items():
    for n in res:
        P=np.vstack([res[m]['P'] for m in res if m!=n]);Y=np.concatenate([[CL.index(y) for y in res[m]['y']] for m in res if m!=n])
        Ts=np.linspace(0.05,1.5,59);T=Ts[int(np.argmax([np.log(soft(P,T)[np.arange(len(Y)),Y]+1e-12).mean() for T in Ts]))]
        cal[(w,n)]=soft(np.array(res[n]['P']),T)
Y={(w,n):np.array([CL.index(y) for y in R[w][n]['y']]) for w in R for n in R[w]}
POS={(w,n):np.array(R[w][n]['pos'])-1 for w in R for n in R[w]}
perms=[rng.permutation(15) for _ in range(NP)]
def permy(w,n,pi):   # same-order position permutation: the same shuffle for all people
    y=Y[(w,n)];pos=POS[(w,n)];lab=np.full(15,-1);lab[pos]=y
    q=lab[pi];q=q[q>=0];return q if len(q)==len(y) else y[rng.permutation(len(y))]
geo=np.load('../../results/08_coordinate_geometry/geom.npz',allow_pickle=True);gCL=list(geo['CL']);ix=[gCL.index(c) for c in CL]
CH=geo['chamfer'][np.ix_(ix,ix)];DD=geo['desc_dist'][np.ix_(ix,ix)]
CURVED=np.array([1,0,1,0,1]);POINTED=np.array([1,0,0,1,0])   # cone cube sphere pyramid cylinder
out={}
# ---------- A. confusion matrices and geometry ----------
iu=np.triu_indices(5,1)
def conf(w,ylab=None):
    M=np.zeros((5,5));c=np.zeros(5)
    for n in R[w]:
        y=Y[(w,n)] if ylab is None else ylab[n]
        for t in range(5): M[t]+=cal[(w,n)][y==t].sum(0);c[t]+=(y==t).sum()
    return M/c[:,None]
A={}
for w in W:
    M=conf(w);S=(M+M.T)/2;off=S[iu]
    r_ch=np.corrcoef(off,-CH[iu])[0,1];r_dd=np.corrcoef(off,-DD[iu])[0,1]
    # Mantel: 120 permutations of the shape labels (exact)
    nch=[np.corrcoef(off,-CH[np.ix_(p,p)][iu])[0,1] for p in map(list,permutations(range(5)))]
    ndd=[np.corrcoef(off,-DD[np.ix_(p,p)][iu])[0,1] for p in map(list,permutations(range(5)))]
    # attribute level: curved/cornered and pointed/not
    def attr(lab):
        e=[];s=[]
        for n in R[w]:
            P=cal[(w,n)];y=lab[n]
            pe=P@CURVED;ps=P@POINTED
            e.append(np.mean((pe>CURVED.mean())==(CURVED[y]==1)));s.append(np.mean((ps>POINTED.mean())==(POINTED[y]==1)))
        return np.mean(e),np.mean(s)
    e0,s0=attr({n:Y[(w,n)] for n in R[w]})
    nul=np.array([attr({n:permy(w,n,pi) for n in R[w]}) for pi in perms[:2000]])
    A[w]=dict(confusion=M.round(3).tolist(),r_chamfer=round(r_ch,3),p_chamfer=float(np.mean(np.array(nch)>=r_ch-1e-12)),
              r_descriptive=round(r_dd,3),p_descriptive=float(np.mean(np.array(ndd)>=r_dd-1e-12)),
              curved_accuracy=round(e0,3),curved_p=float((1+np.sum(nul[:,0]>=e0))/(1+len(nul))),curved_null95=round(float(np.percentile(nul[:,0],95)),3),
              pointed_accuracy=round(s0,3),pointed_p=float((1+np.sum(nul[:,1]>=s0))/(1+len(nul))),pointed_null95=round(float(np.percentile(nul[:,1],95)),3))
    print(w,{k:v for k,v in A[w].items() if k!='confusion'},flush=True)
out['A_confusion_geometry']=A
# ---------- B. within-person similarity across blocks ----------
PAIRS=[('Block 1 · imagery','Block 2 · imagery'),('Block 1 · imagery','Block 3 · imagery'),('Block 2 · imagery','Block 3 · imagery'),
       ('Block 2 · touch to imagery','Block 1 · imagery'),('Block 2 · touch to imagery','Block 3 · imagery'),
       ('Block 1 · word onset','Block 1 · imagery'),('Block 2 · touch','Block 2 · imagery'),('Block 3 · picture onset','Block 3 · imagery'),
       ('Block 3 · picture onset','Block 1 · imagery'),('Block 3 · picture onset','Block 2 · touch to imagery')]
def proto(w,n,y):
    P=cal[(w,n)];return np.array([P[y==t].mean(0) if (y==t).any() else np.full(5,np.nan) for t in range(5)])
def score(a,b,n,yb=None):
    A_=proto(a,n,Y[(a,n)]);B_=proto(b,n,Y[(b,n)] if yb is None else yb)
    C=np.array([[np.corrcoef(A_[i],B_[j])[0,1] if not(np.isnan(A_[i]).any() or np.isnan(B_[j]).any()) else np.nan for j in range(5)] for i in range(5)])
    d=np.nanmean(np.diag(C));o=np.nanmean(C[~np.eye(5,dtype=bool)]);return d-o
B={}
for a,b in PAIRS:
    ns=[n for n in R[a] if n in R[b]]
    s={n:score(a,b,n) for n in ns};m=np.mean(list(s.values()))
    nl=np.array([np.mean([score(a,b,n,permy(b,n,pi)) for n in ns]) for pi in perms[:2000]])
    B[f'{a} ↔ {b}']=dict(n=len(ns),mean=round(float(m),3),p=float((1+np.sum(nl>=m))/(1+len(nl))),null95=round(float(np.percentile(nl,95)),3),person={n:round(float(v),3) for n,v in s.items()})
    print(a,'↔',b,len(ns),round(m,3),B[f'{a} ↔ {b}']['p'],flush=True)
out['B_blocks_between']=B
# ---------- C. groups ----------
k=pd.read_excel('../../participants/participant_information.xlsx');G=dict(zip(k['code'],k['analysis_group']))
def acc(w,n,y=None): y=Y[(w,n)] if y is None else y;return float(np.mean(cal[(w,n)].argmax(1)==y))
C={}
for w in W:
    ns=list(R[w]);grp={}
    for n in ns: grp.setdefault(G.get(n,'?'),[]).append(n)
    obs={g:np.mean([acc(w,n) for n in v]) for g,v in grp.items()}
    nl={g:[] for g in grp}
    for pi in perms[:2000]:
        for g,v in grp.items(): nl[g].append(np.mean([acc(w,n,permy(w,n,pi)) for n in v]))
    vals=[[acc(w,n) for n in v] for v in grp.values() if len(v)>1]
    H=kruskal(*vals) if len(vals)>1 else None
    # label permutation for the group difference
    a=np.array([acc(w,n) for n in ns]);gl=np.array([G.get(n,'?') for n in ns])
    def Hs(lab):
        v=[a[lab==g] for g in np.unique(lab) if (lab==g).sum()>1]
        return kruskal(*v).statistic if len(v)>1 else 0
    h0=Hs(gl);hn=[Hs(rng.permutation(gl)) for _ in range(2000)]
    C[w]={'groups':{g:dict(n=len(grp[g]),accuracy=round(float(obs[g]),3),p_chance=float((1+np.sum(np.array(nl[g])>=obs[g]))/(1+len(nl[g])))) for g in grp},
          'group_diff_H':round(float(h0),3),'group_diff_p':float((1+np.sum(np.array(hn)>=h0))/(1+len(hn)))}
    print(w,C[w],flush=True)
out['C_groups']=C
json.dump(out,open('../../results/18_detailed_analyses/detail_res.json','w'),ensure_ascii=False,indent=1)
