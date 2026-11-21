import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '26_sound_and_ranking'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,pickle,warnings,os;warnings.filterwarnings('ignore')
exec(open('../../src/23_three_groups_cases_connectivity/matrix.py').read().split('rng=np.random.default_rng(3)')[0])
rng=np.random.default_rng(4)
DS=sorted(n for n in OL if 'B1_img' in OL[n] and G(n)=='CLV');GS=sorted(n for n in OL if 'B1_img' in OL[n] and G(n)=='Sighted')
out={}
for cond,lab in [('B1_img','Pattern closeness · B1 (+ congenitally limited, − sighted)')]:
    R=feats(cond);yy={n:np.array([CL.index(q) for q in R[n]['y']]) for n in R}
    for te in R:
        a=[]
        for pool in (DS,GS):
            p=[m for m in pool if m!=te];P=np.mean([hier(R,R,list(rng.choice(p,4,replace=False)),te,'X') for _ in range(10)],0);a.append(np.mean(P.argmax(1)==yy[te]))
        out[te]=a[0]-a[1]
D=pickle.load(open('../../results/26_sound_and_ranking/measure.pkl','rb'))
for n,v in out.items(): D['M'][n][lab]=float(v)
pickle.dump(D,open('../../results/26_sound_and_ranking/measure.pkl','wb'));print({n:round(v,2) for n,v in out.items()})
