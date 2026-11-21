"""Shape generation from every block: nested method selection (training people only) + probability-weighted superquadric + coordinate-based measurement."""
import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '15_shape_generation_every_block'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,pickle,json,os,sys,warnings;warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0,'../../src/09_superquadric_bridge');from sq import PAR
OL=pickle.load(open('../../data/derived/06_features/feats.pkl','rb'));EM=pickle.load(open('../../data/derived/07_things/ours_emb.pkl','rb'))
CL=['cone','cube','sphere','pyramid','cylinder']
BL={'Block 1 · word onset':('B1_erp','B1_word'),'Block 1 · imagery':('B1_img','B1_img'),'Block 2 · touch':('B2_touch','B2_touch'),'Block 2 · imagery':('B2_img','B2_img'),
    'Block 3 · picture onset':('B3_erp','B3_picture'),'Block 3 · picture viewing':('B3_perc','B3_viewing'),'Block 3 · imagery':('B3_img','B3_img'),'Block 2 · touch to imagery':('B2_touch','B2_touch','B2_img','B2_img')}
def z(X):return (X-X.mean(0))/(X.std(0)+1e-9)
def build(a,b):
    R={}
    for n in OL:
        if a in OL[n] and n in EM and b in EM[n]:
            o=OL[n][a];e=EM[n][b];cm=[p for p in o['pos'] if p in set(e['pos'])]
            io=[list(o['pos']).index(p) for p in cm];ie=[list(e['pos']).index(p) for p in cm]
            Xo=z(o['X'][io]);Xe=z(e['Z'][ie]);R[n]=dict(main=Xo,THINGS=Xe,combined=np.hstack([Xo,Xe]),y=o['y'][io],pos=np.array(cm))
    return R
def fitpred(R,tr,te,f):
    c=make_pipeline(StandardScaler(),LogisticRegression(C=0.01,max_iter=3000)).fit(np.vstack([R[m][f] for m in tr]),np.concatenate([R[m]['y'] for m in tr]))
    P=c.predict_proba(R[te][f]);return P[:,[list(c.classes_).index(k) for k in CL]]
out=json.load(open('gen_res.json')) if os.path.exists('gen_res.json') else {}
for name,spec in BL.items():
    a,b=spec[:2]
    if name in out: continue
    R=build(a,b);RT=build(*spec[2:]) if len(spec)>2 else R;names=[n for n in R if n in RT];res={}
    for te in names:
        tr=[m for m in names if m!=te]
        # inner loop: method selection with training people only
        sc={}
        for f in ['main','THINGS','combined']:
            acc=[np.mean(np.array(CL)[fitpred(R,[m for m in tr if m!=v],v,f).argmax(1)]==R[v]['y']) for v in tr]
            sc[f]=np.mean(acc)
        best=max(sc,key=sc.get)
        c=make_pipeline(StandardScaler(),LogisticRegression(C=0.01,max_iter=3000)).fit(np.vstack([R[m][best] for m in tr]),np.concatenate([R[m]['y'] for m in tr]))
        P=c.predict_proba(RT[te][best])[:,[list(c.classes_).index(k) for k in CL]]
        res[te]=dict(P=P.tolist(),y=RT[te]['y'].tolist(),pos=RT[te]['pos'].tolist(),method=best,inner=sc)
    out[name]=res;json.dump(out,open('gen_res.json','w'),ensure_ascii=False)
    acc=np.mean([np.mean(np.array(CL)[np.array(v['P']).argmax(1)]==np.array(v['y'])) for v in res.values()])
    from collections import Counter;print(f'{name:24s} accuracy {acc:.3f} | selected methods {dict(Counter(v['method'] for v in res.values()))}',flush=True)
