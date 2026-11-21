import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '02_timing_alignment'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, glob, json, os, sys
mne.set_log_level('ERROR')
AB=[f'A{i}' for i in range(1,33)]+[f'B{i}' for i in range(1,33)]
out={}
for f in sorted(glob.glob('../../data/raw_bdf/*_block2*.bdf')):
    n=os.path.basename(f)[:-4]
    raw=mne.io.read_raw_bdf(f,preload=True); sf=raw.info['sfreq']
    st=raw.get_data(picks=['Status'])[0].astype(np.int64)&0xFFFF
    on=list((np.nonzero((st[1:]==49)&(st[:-1]!=49))[0]+1)/sf)
    raw.pick(AB); raw.resample(256)
    hi=raw.copy().filter(20,45).get_data(); lo=raw.copy().filter(0.3,3).get_data()
    w=256  # 1 s bins
    nb=hi.shape[1]//w
    eh=np.log(np.median((hi[:,:nb*w].reshape(64,nb,w)**2).mean(2),0))
    hand=np.log(np.median(np.abs(lo[:,:nb*w].reshape(64,nb,w)).mean(2),0))
    np.savez(f'{n}.npz',eh=eh,hand=hand,on=np.array(on))
    print(n,nb,'s',on)
