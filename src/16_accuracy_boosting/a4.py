import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '16_accuracy_boosting'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,json,warnings;warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
exec(open('../../src/15_shape_generation_every_block/gen.py').read().split('out=json.load')[0])
FEAT={'Block 1 · word onset':'THINGS','Block 1 · imagery':'THINGS','Block 2 · touch':'THINGS','Block 2 · imagery':'THINGS','Block 2 · touch to imagery':'THINGS','Block 3 · picture onset':'main','Block 3 · picture viewing':'combined','Block 3 · imagery':'main'}
CURVED={'sphere','cylinder','cone'};POINTED={'cone','pyramid'}
ATTR=np.array([[c in CURVED for c in CL],[c in POINTED for c in CL]],float)  # 2x5
out={}
for name,spec in BL.items():
    a,b=spec[:2];R=build(a,b);RT=build(*spec[2:]) if len(spec)>2 else R;names=[n for n in R if n in RT];f=FEAT[name];res={}
    for te in names:
        tr=[m for m in names if m!=te];X=np.vstack([R[m][f] for m in tr]);y=np.concatenate([R[m]['y'] for m in tr]);sc=StandardScaler().fit(X);Xs=sc.transform(X);Xt=sc.transform(RT[te][f])
        c5=LogisticRegression(C=0.01,max_iter=3000).fit(Xs,y);P5=c5.predict_proba(Xt)[:,[list(c5.classes_).index(q) for q in CL]]
        L=np.log(P5+1e-9)
        for k,S in enumerate([CURVED,POINTED]):
            yb=np.array([v in S for v in y]);cb=LogisticRegression(C=0.01,max_iter=3000,class_weight='balanced').fit(Xs,yb);pb=cb.predict_proba(Xt)[:,1]
            L+=np.log(np.where(ATTR[k][None]==1,pb[:,None],1-pb[:,None])+1e-9)
        P=np.exp(L-L.max(1,keepdims=True));P/=P.sum(1,keepdims=True)
        res[te]=dict(P=P.tolist(),y=RT[te]['y'].tolist(),pos=RT[te]['pos'].tolist())
    out[name]=res;print(f"{name:32s} hierarchical accuracy {np.mean([np.mean(np.array(CL)[np.array(v['P']).argmax(1)]==np.array(v['y'])) for v in res.values()]):.3f}",flush=True)
json.dump(out,open('a4_res.json','w'),ensure_ascii=False)
