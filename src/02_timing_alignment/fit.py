import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '02_timing_alignment'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, glob, json
TOUCH=[0,29,57,84,112,139,166,195,222,251,279,306,335,363,391]
files=sorted(glob.glob('*.npz')); D={f[:-4]:np.load(f) for f in files}
z=lambda e:(e-np.median(e))/(np.std(e)+1e-9)
def template(excl):
    T={'eh':[],'hand':[]}
    for n,d in D.items():
        if n==excl or len(d['on'])==0: continue
        t0=d['on'][-1]
        for k in T:
            e=z(d[k]); s=[e[int(round(t0+x))-5:int(round(t0+x))+28] for x in TOUCH if int(round(t0+x))-5>=0 and int(round(t0+x))+28<=len(e)]
            T[k].append(np.mean(s,0))
    return {k:np.mean(v,0) for k,v in T.items()}
def predict(n,tpl,ncyc=15):
    d=D[n]; L=len(d['eh']); best=[]
    E={k:z(d[k]) for k in ('eh','hand')}
    for off in np.arange(0,L-50,0.25):
        sc=0;cnt=0
        for k in E:
            for x in TOUCH[:ncyc]:
                i=int(round(off+x))
                if i-5<0 or i+28>L: continue
                seg=E[k][i-5:i+28]; sc+=np.corrcoef(seg,tpl[k])[0,1]; cnt+=1
        best.append((sc,off,cnt))
    best.sort(reverse=True); return best
res={}
for n,d in D.items():
    tpl=template(n); b=predict(n,tpl)
    # second-best peak away from best
    sec=next(x for x in b if abs(x[1]-b[0][1])>10)
    true=d['on'][-1] if len(d['on']) else None
    res[n]=dict(prediction=b[0][1],score=round(b[0][0],3),second=round(sec[0],3),cycle=b[0][2],actual=None if true is None else round(float(true),2))
    print(n,res[n],'' if true is None else f'error={b[0][1]-true:+.2f}s')
json.dump(res,open('fit_res.json','w'),ensure_ascii=False)
