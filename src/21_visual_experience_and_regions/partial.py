import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '21_visual_experience_and_regions'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,json,warnings;warnings.filterwarnings('ignore');exec(open('../../src/21_visual_experience_and_regions/mantel.py').read().split('out={}')[0])
from scipy.stats import rankdata
def resid(a,b):a=rankdata(a);b=rankdata(b);B=np.c_[np.ones_like(b),b];return a-B@np.linalg.lstsq(B,a,rcond=None)[0]
rng=np.random.default_rng(1);out={}
for cond,bset,tag in [('B2_img',['alpha'],'B2 imagery · alpha'),('B2_touch',['alpha'],'B2 touch · alpha')]:
    R=feats(cond,bset);ns,S=pair(R);Ss=(S+S.T)/2;iu=np.triu_indices(len(ns),1)
    age=np.array([AGE[n] for n in ns],float);A=-np.abs(age[:,None]-age[None])[iu]
    for vn,V in [('years with typical vision',YRS),('residual vision',VIS)]:
        v=np.array([V[n] for n in ns],float)
        def st(v):return np.corrcoef(resid(Ss[iu],A),resid(-np.abs(v[:,None]-v[None])[iu],A))[0,1]
        r=st(v);nl=[st(v[rng.permutation(len(v))]) for _ in range(5000)]
        out[f'{tag} | {vn} | age controlled']=dict(r=round(float(r),3),p=float((1+np.sum(np.array(nl)>=r))/5001))
    r_age=spearmanr(Ss[iu],A).correlation;out[f'{tag} | age similarity']=round(float(r_age),3)
print(json.dumps(out,ensure_ascii=False,indent=0));json.dump(out,open('partial_res.json','w'),ensure_ascii=False,indent=1)
