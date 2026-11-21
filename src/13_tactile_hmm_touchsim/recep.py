"""Receptor signatures of contact states with TouchSim (index fingertip, 10x10 mm contact area)."""
import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '13_tactile_hmm_touchsim'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import touchsim as ts, numpy as np, json, warnings; warnings.filterwarnings('ignore')
fs=2000; T=0.6; t=np.arange(int(T*fs))/fs
g=np.arange(-5,5.01,0.7); X,Y=np.meshgrid(g,g); loc=np.c_[X.ravel(),Y.ravel()]
ramp=np.clip(t/0.05,0,1)  # press at 50 ms, then hold
D=1.0  # mm indentation depth
def depth(kind,x,y,shift=0.0):
    x=x-shift
    if kind=='flat surface': return np.full_like(x,D)
    if kind=='curved surface': return D-(x**2+y**2)/(2*30.0)            # sphere radius 30 mm
    if kind=='edge': return D-np.abs(x)*np.tan(np.radians(45))      # 90° edge
    if kind=='corner/tip': return D-np.sqrt(x**2+y**2)*np.tan(np.radians(45))
def stim(kind,slide=False):
    if not slide:
        tr=np.clip(depth(kind,loc[:,0],loc[:,1]),0,None)[:,None]*ramp[None]
    else:  # finger slides over the surface at 25 mm/s (edge following / edge crossing)
        tr=np.stack([np.clip(depth(kind,loc[:,0],loc[:,1],shift=-6+25*ti),0,None)*min(ti/0.05,1) for ti in t],1)
    return ts.Stimulus(trace=tr,location=loc,fs=fs,pin_radius=0.35)
ap=ts.affpop_grid(dist=1.0,max_extent=10)
res={}
for kind in ['flat surface','curved surface','edge','corner/tip']:
    for slide in [False,True]:
        r=ap.response(stim(kind,slide)); rate=r.rate().ravel()
        cls=np.array([a.affclass for a in ap.afferents])
        out={c:dict(mean=float(rate[cls==c].mean()),max=float(rate[cls==c].max()),active_ratio=float((rate[cls==c]>5).mean())) for c in ['SA1','RA','PC']}
        key=f"{kind} | {'sliding' if slide else 'pressing'}";res[key]=out
        print(f"{key:24s} "+'  '.join(f"{c}: mean {v['mean']:5.1f} Hz max {v['max']:5.1f}" for c,v in out.items()),flush=True)
json.dump(res,open('recep_res.json','w'),ensure_ascii=False)
