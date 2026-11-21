"""A2 personalisation: other people + 2 repetitions of the test person (25% of the total weight) → test the 3rd repetition; 3 folds."""
import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '16_accuracy_boosting'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,pickle,json,warnings;warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
exec(open('../../src/15_shape_generation_every_block/gen.py').read().split('out=json.load')[0])  # build, z, CL, BL
FEAT={'Block 1 · word onset':'THINGS','Block 1 · imagery':'THINGS','Block 2 · touch':'THINGS','Block 2 · imagery':'THINGS','Block 2 · touch to imagery':'THINGS',
      'Block 3 · picture onset':'main','Block 3 · picture viewing':'combined','Block 3 · imagery':'main'}
def reps(y):
    r=np.zeros(len(y),int);c={}
    for i,v in enumerate(y): r[i]=c.get(v,0);c[v]=r[i]+1
    return r
out={}
for name,spec in BL.items():
    a,b=spec[:2];R=build(a,b);RT=build(*spec[2:]) if len(spec)>2 else R;names=[n for n in R if n in RT];f=FEAT[name];res={}
    for te in names:
        tr=[m for m in names if m!=te];Xo=np.vstack([R[m][f] for m in tr]);yo=np.concatenate([R[m]['y'] for m in tr])
        yt=RT[te]['y'];rt=reps(yt);P=np.zeros((len(yt),5));transfer=len(spec)>2
        for k in range(3):
            te_i=rt==k
            if transfer: Xs,ys=R[te][f],R[te]['y']          # in transfer all of the person's touch data can be used (imagery labels are not used)
            else: Xs,ys=RT[te][f][~te_i],yt[~te_i]
            w=0.25*len(yo)/(0.75*len(ys))
            X=np.vstack([Xo,Xs]);y=np.concatenate([yo,ys]);sw=np.r_[np.ones(len(yo)),np.full(len(ys),w)]
            sc=StandardScaler().fit(X);c=LogisticRegression(C=0.01,max_iter=3000).fit(sc.transform(X),y,sample_weight=sw)
            P[te_i]=c.predict_proba(sc.transform(RT[te][f][te_i]))[:,[list(c.classes_).index(q) for q in CL]]
        res[te]=dict(P=P.tolist(),y=yt.tolist(),pos=RT[te]['pos'].tolist())
    out[name]=res;print(f"{name:32s} adaptive accuracy {np.mean([np.mean(np.array(CL)[np.array(v['P']).argmax(1)]==np.array(v['y'])) for v in res.values()]):.3f}",flush=True)
json.dump(out,open('a2_res.json','w'),ensure_ascii=False)
