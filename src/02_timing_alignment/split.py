import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '02_timing_alignment'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np
exec(open('../../src/02_timing_alignment/aepscan.py').read().split('out={}')[0])
def top(tv,g):
    o=np.argsort(tv)[::-1]; t=[]
    for j in o:
        if all(abs(SH[j]-SH[k])>0.3 for k in t): t.append(j)
        if len(t)==3: break
    return [(float(SH[j]),round(float(tv[j]),2)) for j in t]
A,g,tv,i=scan('../../data/clean_continuous/*_block2*_clean_raw.fif',TOUCH); print('block2 TOUCH beeps only',top(tv,g))
A,g,tv,i=scan('../../data/clean_continuous/*_block2*_clean_raw.fif',IMG); print('block2 IMAGINE beeps only',top(tv,g))
A,g,tv,i=scan('../../data/clean_continuous/*_block1_clean_raw.fif',S1)
for h,idx in [('block1 first half',slice(0,None,2)),('block1 second half',slice(1,None,2))]:
    B=A[idx]; gg=B.mean(0); t2=gg/(B.std(0)/np.sqrt(len(B))); print(h,len(B),'recording',top(t2,gg))
A,g,tv,i=scan('../../data/clean_continuous/*_block1_clean_raw.fif',S1[:7]); print('block1 first 7 sound',top(tv,g))
A,g,tv,i=scan('../../data/clean_continuous/*_block1_clean_raw.fif',S1[7:]); print('block1 last 8 sound',top(tv,g))
