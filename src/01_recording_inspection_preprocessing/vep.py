import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '01_recording_inspection_preprocessing'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, json, glob
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
mne.set_log_level('ERROR')
S3=[0,16,33,51,67,85,102,119,135,153,169,187,203,221,238]
OCC=['O1','Oz','O2','PO7','PO8','POz','PO3','PO4']
E=[];names=[]
for f in sorted(glob.glob('../../data/preprocessing_proc/*_block3.json')):
    j=json.load(open(f))
    if 'error' in j: continue
    on=[t for t,a,b in j['transitions'] if b==49]
    if not on: continue
    n=j['name']; raw=mne.io.read_raw_fif(f'../../data/preprocessing_proc/{n}_raw.fif',preload=True); sf=raw.info['sfreq']
    ev=np.array([[int(round((on[0]+x)*sf)),0,1] for x in S3 if on[0]+x+1<raw.times[-1]])
    ep=mne.Epochs(raw,ev,tmin=-0.2,tmax=0.6,baseline=(-0.2,0),reject=dict(eeg=200e-6),preload=True)
    chs=[c for c in OCC if c not in raw.info['bads']]
    print(n,'epochs',len(ep),'bads',raw.info['bads'])
    if len(ep)<5 or not chs: continue
    e=ep.copy().pick(chs).average().data.mean(0)*1e6; E.append(e); names.append((n,len(ep))); t=ep.times
E=np.array(E); m=E.mean(0); se=E.std(0)/np.sqrt(len(E))
print(len(E),'recording',names)
for a,b in [(0.07,0.13),(0.13,0.22),(0.2,0.35)]:
    w=(t>=a)&(t<b); v=E[:,w].mean(1); print(a,b,'mean',round(v.mean(),2),'t',round(v.mean()/(v.std()/np.sqrt(len(v))),2))
plt.figure(figsize=(8,3.5)); plt.fill_between(t*1000,m-se,m+se,alpha=.25,color='#1f8a8a'); plt.plot(t*1000,m,color='#1f8a8a')
plt.axvline(0,color='k',lw=.8); plt.axhline(0,color='gray',lw=.5)
plt.title(f'Block 3 – visual cortex response when the shape appears on screen ({len(E)} recordings)'); plt.xlabel('After planned image onset (ms)'); plt.ylabel('Occipital mean (µV)')
plt.tight_layout(); plt.savefig('block3_visual_response.png',dpi=130)
