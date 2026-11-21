import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '16_accuracy_boosting'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,pickle,json,os,sys,warnings;warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
OL=pickle.load(open('../../data/derived/06_features/feats.pkl','rb'));EM=pickle.load(open('../../data/derived/07_things/ours_emb.pkl','rb'));RI=pickle.load(open('../../data/derived/06_features/feats_riem.pkl','rb'))
CL=['cone','cube','sphere','pyramid','cylinder']
BL={'Block 1 · imagery':('B1_img','B1_img','B1_img'),'Block 2 · touch':('B2_touch','B2_touch','B2_touch'),'Block 2 · imagery':('B2_img','B2_img','B2_img'),
    'Block 3 · picture viewing':('B3_perc','B3_viewing','B3_perc'),'Block 3 · imagery':('B3_img','B3_img','B3_img')}
def z(X):return (X-X.mean(0))/(X.std(0)+1e-9)
def build(a,b,c):
    R={}
    for n in OL:
        if a in OL[n] and n in EM and b in EM[n] and n in RI and c in RI[n]:
            o,e,r=OL[n][a],EM[n][b],RI[n][c];cm=[p for p in o['pos'] if p in set(e['pos']) and p in set(r['pos'])]
            ix=lambda d:[list(d['pos']).index(p) for p in cm]
            Xo,Xe,Xr=z(o['X'][ix(o)]),z(e['Z'][ix(e)]),z(r['X'][ix(r)])
            R[n]=dict(main=Xo,THINGS=Xe,riemann=Xr,THINGS_riemann=np.hstack([Xe,Xr]),combined=np.hstack([Xo,Xe]),y=o['y'][ix(o)],pos=np.array(cm))
    return R
FS=['main','THINGS','riemann','THINGS_riemann','combined']
def fp(R,tr,te,f):
    c=make_pipeline(StandardScaler(),LogisticRegression(C=0.01,max_iter=3000)).fit(np.vstack([R[m][f] for m in tr]),np.concatenate([R[m]['y'] for m in tr]))
    return c.predict_proba(R[te][f])[:,[list(c.classes_).index(k) for k in CL]]
out=json.load(open('a1_res.json')) if os.path.exists('a1_res.json') else {}
for name,spec in BL.items():
    if name in out: continue
    R=build(*spec);names=list(R);res={}
    # inner-loop speed-up: the held-out predictions of candidate methods are computed once for each training person
    for te in names:
        tr=[m for m in names if m!=te];sc={}
        for f in FS:
            sc[f]=np.mean([np.mean(np.array(CL)[fp(R,[m for m in tr if m!=v],v,f).argmax(1)]==R[v]['y']) for v in tr[::2]])  # inner loop: half of the training people
        best=max(sc,key=sc.get);P=fp(R,tr,te,best)
        res[te]=dict(P=P.tolist(),y=R[te]['y'].tolist(),pos=R[te]['pos'].tolist(),method=best)
    out[name]=res;json.dump(out,open('a1_res.json','w'),ensure_ascii=False)
    from collections import Counter
    print(name,'accuracy',round(np.mean([np.mean(np.array(CL)[np.array(v['P']).argmax(1)]==np.array(v['y'])) for v in res.values()]),3),dict(Counter(v['method'] for v in res.values())),flush=True)
