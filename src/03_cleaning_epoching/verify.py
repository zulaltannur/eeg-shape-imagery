import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '03_cleaning_epoching'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne,glob,numpy as np,json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
mne.set_log_level('ERROR')
FC=['Fz','FCz','Cz','FC1','FC2','C1','C2']; OCC=['O1','Oz','O2','PO7','PO8','POz','PO3','PO4']
def ev_erp(f,chs,lo=-0.2,hi=0.6,offsets=(0,)):
    ep=mne.read_epochs(f,preload=True)
    out=[]
    for o in offsets:
        e=ep.copy().crop(lo+o,hi+o) if o else ep.copy().crop(lo,hi)
        x=e.get_data(picks=chs).mean(1)*1e6; x=x-x[:,:int(0.2*256)].mean(1,keepdims=True)
        keep=np.ptp(x,1)<100; out.append(x[keep].mean(0))
    return np.mean(out,0)
res={}
fig,ax=plt.subplots(1,3,figsize=(15,3.8))
specs=[('block3 visual perception',glob.glob('../../data/clean_epochs/epo/*block3_visual_perception-epo.fif'),OCC,'Occipital'),
       ('block2 touch/imagine beeps',glob.glob('../../data/clean_epochs/epo/*block2*_touch-epo.fif')+glob.glob('../../data/clean_epochs/epo/*block2*_imagery-epo.fif'),FC,'Fronto-central'),
       ('block1 sounds',glob.glob('../../data/clean_epochs/epo/*block1_sound_and_imagery-epo.fif'),FC,'Fronto-central')]
for k,(lab,files,chs,reg) in enumerate(specs):
    E=np.array([ev_erp(f,chs) for f in sorted(files) if 'P19' not in f])
    t=np.linspace(-0.2,0.6,E.shape[1]); m=E.mean(0); se=E.std(0)/np.sqrt(len(E))
    for a,b in [(0.07,0.13),(0.13,0.22),(0.2,0.35)]:
        w=(t>=a)&(t<b); v=E[:,w].mean(1); print(lab,a,b,'mean',round(v.mean(),2),'t',round(v.mean()/(v.std()/np.sqrt(len(v))),2))
    ax[k].fill_between(t*1000,m-se,m+se,alpha=.25,color='#1f8a8a'); ax[k].plot(t*1000,m,color='#1f8a8a'); ax[k].axvline(0,color='k',lw=.8); ax[k].axhline(0,color='gray',lw=.5)
    ax[k].set_title(f'{lab} ({len(E)} recordings)',fontsize=10); ax[k].set_xlabel('After planned onset (ms)'); ax[k].set_ylabel(f'{reg} (µV)')
plt.tight_layout(); plt.savefig('verification_clean.png',dpi=120)
