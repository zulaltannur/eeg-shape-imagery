import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '25_stability_and_P04'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Within-person shape-specific stability (how similar the repetitions of the same shape are) versus the model's ability to read that person
import numpy as np,json,pickle
from scipy.stats import spearmanr
CL=['cone','cube','sphere','pyramid','cylinder'];OL=pickle.load(open('../../data/derived/06_features/feats.pkl','rb'));R=json.load(open('../../results/16_accuracy_boosting/final_res.json'))
W={'B1_img':'Block 1 · imagery','B2_img':'Block 2 · imagery','B2_touch':'Block 2 · touch','B3_img':'Block 3 · imagery','B3_perc':'Block 3 · picture viewing'}
def stab(X,y):
    X=(X-X.mean(0))/(X.std(0)+1e-9);occ=np.zeros(len(y),int)
    for t in set(y): occ[y==t]=np.arange((y==t).sum())
    s=[];d=[]
    for i in range(len(y)):
        for j in range(i+1,len(y)):
            if occ[i]==occ[j]: continue
            r=np.corrcoef(X[i],X[j])[0,1];(s if y[i]==y[j] else d).append(r)
    return np.mean(s)-np.mean(d)
def acc(w,n):P=np.array(R[w][n]['P']);y=np.array([CL.index(q) for q in R[w][n]['y']]);return float(np.mean(P.argmax(1)==y))
rng=np.random.default_rng(0);out={};pool_s=[];pool_a=[];pool_w=[]
for c,w in W.items():
    ns=[n for n in OL if c in OL[n] and n in R[w]];S=np.array([stab(OL[n][c]['X'],np.array(OL[n][c]['y'])) for n in ns]);A=np.array([acc(w,n) for n in ns])
    r=spearmanr(S,A).correlation;nl=[spearmanr(S,rng.permutation(A)).correlation for _ in range(10000)]
    out[w]=dict(n=len(ns),rho=round(float(r),3),p=float((1+np.sum(np.array(nl)>=r))/10001),person={n:[round(float(s),4),round(float(a),3)] for n,s,a in zip(ns,S,A)})
    print(w,len(ns),round(r,3),round(out[w]['p'],4),flush=True)
    pool_s+=list((S-S.mean())/S.std());pool_a+=list((A-A.mean())/A.std());pool_w+=[w]*len(ns)
# all windows together (standardised within window), permutation within window
ps=np.array(pool_s);pa=np.array(pool_a);pw=np.array(pool_w);r=spearmanr(ps,pa).correlation;nl=[]
for _ in range(10000):
    q=pa.copy()
    for w in set(pw): m=pw==w;q[m]=rng.permutation(q[m])
    nl.append(spearmanr(ps,q).correlation)
out['all']=dict(n=len(ps),rho=round(float(r),3),p=float((1+np.sum(np.array(nl)>=r))/10001));print('all',out['all'])
json.dump(out,open('stability_res.json','w'),ensure_ascii=False,indent=1)
