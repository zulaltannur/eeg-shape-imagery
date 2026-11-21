import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '16_accuracy_boosting'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,pickle,json,warnings;warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
F=pickle.load(open('a5_feats.pkl','rb'));CL=['cone','cube','sphere','pyramid','cylinder']
CURVED={'sphere','cylinder','cone'};POINTED={'cone','pyramid'};ATTR=np.array([[c in CURVED for c in CL],[c in POINTED for c in CL]],float)
B={'Block 1 · word onset':('B1_word','B1_word'),'Block 1 · imagery':('B1_img','B1_img'),'Block 2 · touch':('B2_touch','B2_touch'),'Block 2 · imagery':('B2_img','B2_img'),
   'Block 2 · touch to imagery':('B2_touch','B2_img'),'Block 3 · picture onset':('B3_picture','B3_picture'),'Block 3 · picture viewing':('B3_viewing','B3_viewing'),'Block 3 · imagery':('B3_img','B3_img')}
def z(X):return (X-X.mean(0))/(X.std(0)+1e-9)
out={}
for layer in ['H','H1']:
    for name,(a,b) in B.items():
        names=[n for n in F if a in F[n] and b in F[n]];res={}
        for te in names:
            tr=[m for m in names if m!=te];X=np.vstack([z(F[m][a][layer]) for m in tr]);y=np.concatenate([F[m][a]['y'] for m in tr]);sc=StandardScaler().fit(X);Xs=sc.transform(X);Xt=sc.transform(z(F[te][b][layer]))
            c5=LogisticRegression(C=0.01,max_iter=3000).fit(Xs,y);L=np.log(c5.predict_proba(Xt)[:,[list(c5.classes_).index(q) for q in CL]]+1e-9)
            for k,S in enumerate([CURVED,POINTED]):
                cb=LogisticRegression(C=0.01,max_iter=3000,class_weight='balanced').fit(Xs,np.array([v in S for v in y]));pb=cb.predict_proba(Xt)[:,1]
                L+=np.log(np.where(ATTR[k][None]==1,pb[:,None],1-pb[:,None])+1e-9)
            P=np.exp(L-L.max(1,keepdims=True));P/=P.sum(1,keepdims=True)
            res[te]=dict(P=P.tolist(),y=F[te][b]['y'].tolist(),pos=F[te][b]['pos'].tolist())
        out[f'{layer}|{name}']=res
        print(f"{layer:3s} {name:32s} accuracy {np.mean([np.mean(np.array(CL)[np.array(v['P']).argmax(1)]==np.array(v['y'])) for v in res.values()]):.3f}",flush=True)
json.dump(out,open('a5_res.json','w'),ensure_ascii=False)
