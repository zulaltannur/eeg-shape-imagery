import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '09_superquadric_bridge'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,pickle,json,warnings;warnings.filterwarnings('ignore')
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from joblib import Parallel,delayed
from sq import PAR,CL
OL=pickle.load(open('../../data/derived/06_features/feats.pkl','rb'));EM=pickle.load(open('../../data/derived/07_things/ours_emb.pkl','rb'))
COND={'Block 1 · word onset':('B1_erp','B1_word'),'Block 1 · imagery':('B1_img','B1_img'),'Block 2 · touch':('B2_touch','B2_touch'),
      'Block 2 · imagery':('B2_img','B2_img'),'Block 3 · picture onset':('B3_erp','B3_picture'),'Block 3 · picture viewing':('B3_perc','B3_viewing'),'Block 3 · imagery':('B3_img','B3_img')}
def z(X):return (X-X.mean(0))/(X.std(0)+1e-9)
D={}
for nm,(a,b) in COND.items():
    for n in OL:
        if a in OL[n] and n in EM and b in EM[n]:
            o=OL[n][a];e=EM[n][b];common=[p for p in o['pos'] if p in set(e['pos'])]
            io=[list(o['pos']).index(p) for p in common];ie=[list(e['pos']).index(p) for p in common]
            D.setdefault(nm,{})[n]=dict(X=np.hstack([z(o['X'][io]),z(e['Z'][ie])]),y=o['y'][io],pos=np.array(common))
def seqs(nm):
    S={n:dict(zip(d['pos'].tolist(),d['y'].tolist())) for n,d in D[nm].items()};com={}
    for n,m in S.items():
        if not(n=='P23' and 'Block 2' in nm):[com.setdefault(p,l) for p,l in m.items()]
    for n in S:
        bb=dict(com) if not(n=='P23' and 'Block 2' in nm) else {15:'pyramid'};bb.update(S[n]);S[n]=bb
    return S
T=np.array([PAR[c] for c in CL])
def run(nm,perm=None,S=None):
    dd=D[nm];pred={};acc=[]
    for n in dd:
        tr=[m for m in dd if m!=n];X=np.vstack([dd[m]['X'] for m in tr])
        ys=np.concatenate([dd[m]['y'] if perm is None else np.array([S[m][perm[p-1]] for p in dd[m]['pos']]) for m in tr])
        sc=StandardScaler().fit(X);r=RidgeCV(alphas=[1e2,1e3,1e4,1e5]).fit(sc.transform(X),np.array([PAR[c] for c in ys]))
        P=np.clip(r.predict(sc.transform(dd[n]['X'])),0,1);pred[n]=P
        acc.append(np.mean(np.array(CL)[((P[:,None]-T[None])**2).sum(-1).argmin(1)]==dd[n]['y']))
    return float(np.mean(acc)),pred
rng=np.random.default_rng(5);PER=[rng.permutation(np.arange(1,16)) for _ in range(200)];out={}
for nm in COND:
    a,pred=run(nm);S=seqs(nm);nl=np.array(Parallel(n_jobs=2)(delayed(lambda p:run(nm,p,S)[0])(p) for p in PER));p=(np.sum(nl>=a)+1)/201
    mean={c:np.vstack([pred[n][D[nm][n]['y']==c] for n in pred]).mean(0).tolist() for c in CL}
    out[nm]=dict(acc=a,p=float(p),null95=float(np.percentile(nl,95)),mean=mean,pred={n:pred[n].tolist() for n in pred},y={n:D[nm][n]['y'].tolist() for n in pred})
    print(f'{nm:24s} n={len(pred)} nearest-shape accuracy {a:.3f} (chance 95% {np.percentile(nl,95):.3f}) p={p:.3f}',flush=True)
json.dump(out,open('sq_res.json','w'),ensure_ascii=False)
