"""A3 training blocks together (shared THINGS representation space): (i) all blocks, (ii) all blocks except the test block."""
import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '16_accuracy_boosting'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,pickle,json,warnings;warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
EM=pickle.load(open('../../data/derived/07_things/ours_emb.pkl','rb'));CL=['cone','cube','sphere','pyramid','cylinder']
B={'B1_word':'Block 1 · word onset','B1_img':'Block 1 · imagery','B2_touch':'Block 2 · touch','B2_img':'Block 2 · imagery','B3_picture':'Block 3 · picture onset','B3_viewing':'Block 3 · picture viewing','B3_img':'Block 3 · imagery'}
def z(X):return (X-X.mean(0))/(X.std(0)+1e-9)
names=list(EM);out={'all':{},'others':{}}
for mode in ['all','others']:
    for tb,tn in B.items():
        res={}
        for te in names:
            if tb not in EM[te]: continue
            src=[b for b in B if (mode=='all' or b!=tb)]
            X=[];y=[]
            for m in names:
                if m==te: continue
                for b in src:
                    if b in EM[m]: X.append(z(EM[m][b]['Z']));y.append(EM[m][b]['y'])
            X=np.vstack(X);y=np.concatenate(y);sc=StandardScaler().fit(X)
            c=LogisticRegression(C=0.01,max_iter=3000).fit(sc.transform(X),y)
            P=c.predict_proba(sc.transform(z(EM[te][tb]['Z'])))[:,[list(c.classes_).index(q) for q in CL]]
            res[te]=dict(P=P.tolist(),y=EM[te][tb]['y'].tolist(),pos=EM[te][tb]['pos'].tolist())
        out[mode][tn]=res
        print(f"{mode:10s} {tn:24s} accuracy {np.mean([np.mean(np.array(CL)[np.array(v['P']).argmax(1)]==np.array(v['y'])) for v in res.values()]):.3f}",flush=True)
json.dump(out,open('a3_res.json','w'),ensure_ascii=False)
