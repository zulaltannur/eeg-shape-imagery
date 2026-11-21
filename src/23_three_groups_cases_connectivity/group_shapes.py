import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '23_three_groups_cases_connectivity'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Shapes in mind with the group pattern: B2 imagining what was touched, alpha
import numpy as np,json,pickle,warnings,os,sys;warnings.filterwarnings('ignore')
exec(open('../../src/23_three_groups_cases_connectivity/matrix.py').read().split('rng=np.random.default_rng(3)')[0])
sys.path.insert(0,'../../src/09_superquadric_bridge');from sq import PAR,draw
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
PR=np.array([PAR[c] for c in CL]);rng=np.random.default_rng(5)
R=feats('B2_img');yy={n:np.array([CL.index(q) for q in R[n]['y']]) for n in R};grp={g:sorted(n for n in R if G(n)==g) for g in ['CLV','LVL','Sighted']}
def predict(te,pool,m=4,k=20):
    pool=[x for x in pool if x!=te];return np.mean([hier(R,R,list(rng.choice(pool,min(m,len(pool)),replace=False)),te,'X') for _ in range(k)],0)
COLS=[('Congenitally limited vision\nwith own group pattern','CLV','CLV'),('Congenitally limited vision\nwith the sighted pattern','CLV','Sighted'),
      ('Typical vision then lost\nwith the sighted pattern','LVL','Sighted'),('Sighted\nwith the sighted pattern','Sighted','Sighted')]
P={};out={}
for lab,tg,sg in COLS:
    P[lab]={n:predict(n,grp[sg]) for n in grp[tg]}
# temperature: joint log-likelihood of all cells
pickle.dump((P,yy),open('../../results/23_three_groups_cases_connectivity/groupP.pkl','wb'))
allP=np.vstack([p for d in P.values() for p in d.values()]);allY=np.concatenate([yy[n] for d in P.values() for n in d])
def soft(P,T):L=np.log(P+1e-9)/T;E=np.exp(L-L.max(1,keepdims=True));return E/E.sum(1,keepdims=True)
T=0.3   # display sharpness: the same in all columns (in calibration the best T hit the bound (1.5), i.e. the probabilities are already flat)
fig=plt.figure(figsize=(2.3*5,2.4*5+1.4))
for i,c in enumerate(CL):
    draw(fig.add_subplot(5,5,i*5+1,projection='3d'),PAR[c],('TRUE SHAPE\n' if i==0 else '')+c)
    for j,(lab,tg,sg) in enumerate(COLS):
        Ps=np.vstack([soft(P[lab][n],T)[yy[n]==i] for n in P[lab]]);pm=(Ps@PR).mean(0);pr=Ps.mean(0)
        acc=np.mean([np.mean(P[lab][n].argmax(1)==yy[n]) for n in P[lab]])
        hit=np.mean(np.concatenate([P[lab][n][yy[n]==i].argmax(1)==i for n in P[lab]]))
        out.setdefault(lab,{'overall_accuracy':round(float(acc),3)})[c]=dict(hit_rate=round(float(hit),3),most_likely=CL[int(pr.argmax())])
        draw(fig.add_subplot(5,5,i*5+j+2,projection='3d'),pm,(f"{lab}\noverall accuracy %{acc*100:.0f}\n\n" if i==0 else '')+f"{c}: %{hit*100:.0f} hit_rate")
fig.suptitle("Imagining what was touched (alpha rhythm): shapes in each group's mind, by which group's brain pattern they are read with\n(the model never saw the person being read; each training set 4 people, 20 repetitions; the same sharpness T=0.3 in all columns of the drawing)",fontsize=11)
plt.savefig('../../results/23_three_groups_cases_connectivity/group_shapes.png',dpi=95,bbox_inches='tight');print(T,json.dumps(out,ensure_ascii=False))
json.dump(out,open('../../results/23_three_groups_cases_connectivity/group_shapes.json','w'),ensure_ascii=False,indent=1)
