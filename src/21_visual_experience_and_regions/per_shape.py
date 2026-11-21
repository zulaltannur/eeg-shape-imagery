import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '21_visual_experience_and_regions'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Pattern specific to congenitally low vision (B2 imagery, alpha): which shapes and attributes?
import numpy as np,json,sys,os
sys.argv=['tf.py','B2_img','alpha','0'];src=open('../../src/20_time_frequency/tf.py').read().split("f=f'../../results/20_time_frequency/res_")[0];exec(src)
rng=np.random.default_rng(11);EG=np.array([1,0,1,0,1]);SV=np.array([1,0,0,1,0])
res={'same group':{},'sighted':{}};Pd={}
for te in D:
    cd=[m for m in D if m!=te]
    Pd[te]=(np.mean([P(tuple(rng.choice(cd,4,replace=False)),te) for _ in range(20)],0),np.mean([P(tuple(rng.choice(S,4,replace=False)),te) for _ in range(20)],0))
out={}
for j,src_ in enumerate(['same group','sighted']):
    sh={c:[] for c in CL};eg=[];sv=[]
    for te,pp in Pd.items():
        p=pp[j];y=yy[te];pr=p.argmax(1)
        for t,c in enumerate(CL): sh[c]+=list(pr[y==t]==t)
        eg+=list(((p@EG)>0.6)==(EG[y]==1));sv+=list(((p@SV)>0.4)==(SV[y]==1))
    out[src_]=dict(shape={c:round(float(np.mean(v)),3) for c,v in sh.items()},curved=round(float(np.mean(eg)),3),pointed=round(float(np.mean(sv)),3))
    print(src_,out[src_])
# confusion matrix (same group)
M=np.zeros((5,5))
for te,pp in Pd.items():
    for t in range(5): M[t]+=pp[0][yy[te]==t].sum(0)
M=M/M.sum(1,keepdims=True);out['confusion_same_group']=M.round(3).tolist();print(np.round(M*100))
json.dump(out,open('../../results/21_visual_experience_and_regions/per_shape_res.json','w'),ensure_ascii=False,indent=1)
