import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '16_accuracy_boosting'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,json,os,warnings;warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
exec(open('../../src/15_shape_generation_every_block/gen.py').read().split('out=json.load')[0])
CURVED={'sphere','cylinder','cone'};POINTED={'cone','pyramid'};ATTR=np.array([[c in CURVED for c in CL],[c in POINTED for c in CL]],float)
def hier(R,RT,tr,te,f):
    X=np.vstack([R[m][f] for m in tr]);y=np.concatenate([R[m]['y'] for m in tr]);sc=StandardScaler().fit(X);Xs=sc.transform(X);Xt=sc.transform(RT[te][f])
    c5=LogisticRegression(C=0.01,max_iter=2000).fit(Xs,y);L=np.log(c5.predict_proba(Xt)[:,[list(c5.classes_).index(q) for q in CL]]+1e-9)
    for k,S in enumerate([CURVED,POINTED]):
        cb=LogisticRegression(C=0.01,max_iter=2000,class_weight='balanced').fit(Xs,np.array([v in S for v in y]));pb=cb.predict_proba(Xt)[:,1]
        L+=np.log(np.where(ATTR[k][None]==1,pb[:,None],1-pb[:,None])+1e-9)
    P=np.exp(L-L.max(1,keepdims=True));return P/P.sum(1,keepdims=True)
out=json.load(open('final_res.json')) if os.path.exists('final_res.json') else {}
import time;T0=time.time()
for name,spec in BL.items():
    if name in out: continue
    if time.time()-T0>480: break
    a,b=spec[:2];R=build(a,b);RT=build(*spec[2:]) if len(spec)>2 else R;names=[n for n in R if n in RT];res={}
    for te in names:
        tr=[m for m in names if m!=te];sc={}
        for f in ['main','THINGS','combined']:   # inner loop: training people only (leaving half out)
            sc[f]=np.mean([np.mean(np.array(CL)[hier(R,RT,[m for m in tr if m!=v],v,f).argmax(1)]==RT[v]['y']) for v in tr[::2]])
        best=max(sc,key=sc.get);P=hier(R,RT,tr,te,best)
        res[te]=dict(P=P.tolist(),y=RT[te]['y'].tolist(),pos=RT[te]['pos'].tolist(),method=best)
    out[name]=res;json.dump(out,open('final_res.json','w'),ensure_ascii=False)
    from collections import Counter
    print(f"{name:32s} accuracy {np.mean([np.mean(np.array(CL)[np.array(v['P']).argmax(1)]==np.array(v['y'])) for v in res.values()]):.3f} {dict(Counter(v['method'] for v in res.values()))}",flush=True)
