import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '12_active_exploration'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,pickle,json,warnings;warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel,delayed
D=pickle.load(open('active_feats.pkl','rb'))
def z(X):return (X-X.mean(0))/(X.std(0)+1e-9)
def within(n,perm=None):
    X=z(D[n]['X']);y=D[n]['y'].copy();tri=D[n]['tri'];rep=D[n]['rep']
    if perm is not None:  # shuffle shape names within each round (repetition): balance is preserved
        r=np.random.default_rng(int(perm));sh={}
        for k in np.unique(rep):
            ts=np.unique(tri[rep==k]);lab=np.array([y[tri==t][0] for t in ts]);pl=r.permutation(lab)
            sh.update(dict(zip(ts,pl)))
        y=np.array([sh[t] for t in tri])
    hits=[]
    for k in np.unique(rep):
        te=rep==k;tr=~te
        c=make_pipeline(StandardScaler(),LogisticRegression(C=0.01,max_iter=2000,class_weight='balanced')).fit(X[tr],y[tr])
        P=c.predict_proba(X[te]);cls=c.classes_
        for t in np.unique(tri[te]):
            m=tri[te]==t;hits.append(cls[P[m].mean(0).argmax()]==y[te][m][0])
    return float(np.mean(hits))
W={n:within(n) for n in D};mw=np.mean(list(W.values()))
print('true mean',round(mw,3),{k:round(v,2) for k,v in sorted(W.items(),key=lambda x:-x[1])},flush=True)
rng=np.random.default_rng(1)
null=Parallel(n_jobs=2)(delayed(lambda pm:np.mean([within(n,pm) for n in D]))(int(rng.integers(1e9))) for _ in range(100))
p=(np.sum(np.array(null)>=mw)+1)/101
print(f'WITHIN-PERSON active exploration (1-s windows): mean {mw:.3f} chance mean {np.mean(null):.3f} 95% {np.percentile(null,95):.3f} p={p:.3f}')
json.dump(dict(acc=mw,p=p,null95=float(np.percentile(null,95)),person=W),open('active_res.json','w'),ensure_ascii=False)
