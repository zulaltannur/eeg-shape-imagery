import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '11_eeg3d_check'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from train3d import load, OBJ, emb
rng=np.random.default_rng(0)
res=[]
for s in ['sub01','sub02','sub03','sub04']:
    X,nm=load(s); F=X[:,:,10:].reshape(len(X),X.shape[1],9,10).mean(-1).reshape(len(X),-1)  # 0.1-1 s, 100 ms bins
    te=np.array([n[-2:] in('08','09') for n in nm]); Y=np.stack([OBJ[n] for n in nm])
    sc=StandardScaler().fit(F[~te]); r=RidgeCV(alphas=[1e2,1e3,1e4,1e5]).fit(sc.transform(F[~te]),Y[~te])
    P=r.predict(sc.transform(F[te]))
    objs=sorted(set(nm[te])); 
    # object level: mean of repetitions
    Pm=np.stack([P[nm[te]==o].mean(0) for o in objs]); T=np.stack([OBJ[o] for o in objs])
    C=np.corrcoef(Pm,T)[:len(objs),len(objs):]; rank=np.mean([(C[i]>C[i,i]).sum()+1 for i in range(len(objs))])
    null=[np.mean([(C[i][p]>C[i,p[i]]).sum()+1 for i in range(len(objs))]) for p in [rng.permutation(len(objs)) for _ in range(200)]]
    dimr=[np.corrcoef(Pm[:,k],T[:,k])[0,1] for k in range(T.shape[1])]
    print(s,f'{len(objs)} unseen objects: mean rank of the correct object {rank:.1f} (chance ~{np.mean(null):.1f}, 5% bound {np.percentile(null,5):.1f}) | geometry dimension correlations (first 4):',np.round(dimr[:4],2),flush=True)
