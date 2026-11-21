import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '02_timing_alignment'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, glob
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
TOUCH=[0,29,57,84,112,139,166,195,222,251,279,306,335,363,391]
C={'eh':[],'hand':[]}
for f in sorted(glob.glob('*.npz')):
    d=np.load(f)
    if len(d['on'])==0: continue
    t0=d['on'][-1]
    for k in ('eh','hand'):
        e=(d[k]-np.median(d[k]))/(np.std(d[k])+1e-9); segs=[]
        for x in TOUCH:
            i=int(round(t0+x))
            if i-5>=0 and i+28<=len(e): segs.append(e[i-5:i+28])
        C[k].append(np.mean(segs,0))
t=np.arange(-5,28)
fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
for a,(k,lab) in zip(ax,[('eh','Muscle power 20–45 Hz'),('hand','Slow movement 0.3–3 Hz')]):
    M=np.array(C[k]); m=M.mean(0); se=M.std(0)/np.sqrt(len(M))
    a.fill_between(t,m-se,m+se,alpha=.25,color='#1f8a8a'); a.plot(t,m,color='#1f8a8a')
    for x,l in [(0,'touch beep'),(13,'imagine beep')]: a.axvline(x,color='k',lw=.8); a.text(x+.2,a.get_ylim()[1]*0.9 if a.get_ylim()[1]>0 else 0,l,fontsize=8)
    a.set_ylabel(lab+' (z)')
    print(k,' '.join(f'{v:.2f}' for v in m))
ax[1].set_xlabel('After the touch beep (s)'); ax[0].set_title(f'Block 2 – {len(C["eh"])} recordings: cycle mean of the movement/muscle trace')
plt.tight_layout(); plt.savefig('block2_movement_cycle.png',dpi=120)
