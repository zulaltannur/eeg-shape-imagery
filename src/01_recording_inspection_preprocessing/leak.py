import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '01_recording_inspection_preprocessing'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, sys
from scipy.signal import spectrogram
mne.set_log_level('ERROR')
f=sys.argv[1]
raw=mne.io.read_raw_bdf(f,preload=True); sf=raw.info['sfreq']
st=raw.get_data(picks=['Status'])[0].astype(np.int64)&0xFFFF
t0=(np.nonzero((st[1:]==49)&(st[:-1]!=49))[0][0]+1)/sf
AB=[f'A{i}' for i in range(1,33)]+[f'B{i}' for i in range(1,33)]
X=raw.get_data(picks=AB)
x=X.mean(0)  # common signal (pickup)
fr,tt,S=spectrogram(x,fs=sf,nperseg=1024,noverlap=512)
# find bands with strongest temporal variance (ratio of 95th/median) above 100 Hz
P=10*np.log10(S+1e-30); hi=fr>80
present=np.percentile(P,99,axis=1)-np.median(P,axis=1)
top=np.argsort(np.where(hi,present,-1))[-8:]
print('peaky bands Hz',[ (round(fr[i]),round(present[i],1)) for i in top])
b=top[-1]; env=P[b]; thr=np.median(env)+0.5*(np.percentile(env,99)-np.median(env))
on=tt[np.nonzero((env[1:]>thr)&(env[:-1]<=thr))[0]+1]
print('t0',round(t0,3),'band',round(fr[b]),'onsets rel t0:',np.round(on-t0,2)[:60])
