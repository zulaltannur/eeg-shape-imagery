import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '01_recording_inspection_preprocessing'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, json, glob
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
mne.set_log_level('ERROR')
S1=[0,12,23,33,43,50,64,71,80,90,100,112,123,135,143]
TOUCH=[0,29,57,84,112,139,166,195,222,251,279,306,335,363,391]; IMG=[13,42,70,97,125,152,179,208,235,264,292,319,348,376,404]
CH=('Fz','FCz','Cz','FC1','FC2','C1','C2','F1','F2')
TMIN,TMAX=-0.5,6.0
def get(name,ons):
    j=json.load(open(f'../../data/preprocessing_proc/{name}.json')); on=[t for t,a,b in j['transitions'] if b==49]
    if not on: return None
    t0=on[0]; raw=mne.io.read_raw_fif(f'../../data/preprocessing_proc/{name}_raw.fif',preload=True); sf=raw.info['sfreq']
    chs=[c for c in CH if c not in raw.info['bads']]; X=raw.get_data(picks=chs).mean(0)*1e6
    n=int((TMAX-TMIN)*sf); out=np.full((len(ons),n),np.nan)
    for k,x in enumerate(ons):
        i=int(round((t0+x+TMIN)*sf))
        if i<0 or i+n>len(X): continue
        seg=X[i:i+n]; seg=seg-seg[:int(0.3*sf)].mean()
        if np.abs(seg).max()<150: out[k]=seg
    return out
res={}
for blk,ons,pat in [('block1',S1,'../../data/preprocessing_proc/*_block1.json'),('block2',sorted(TOUCH+IMG),'../../data/preprocessing_proc/*_block2*.json')]:
    A=[]
    for f in sorted(glob.glob(pat)):
        if 'error' in json.load(open(f)): continue
        r=get(f.split('/')[-1][:-5],ons)
        if r is not None: A.append(r)
    A=np.array(A); res[blk]=A; print(blk,A.shape)
t=np.arange(A.shape[2])/256+TMIN
np.save('perevent.npy',{'t':t,**res},allow_pickle=True)
fig,ax=plt.subplots(1,2,figsize=(14,8))
for k,blk in enumerate(['block1','block2']):
    A=res[blk]; M=np.nanmean(A,0)
    from scipy.ndimage import gaussian_filter1d
    M=gaussian_filter1d(M,4,axis=1)
    im=ax[k].imshow(M,aspect='auto',extent=[t[0],t[-1],len(M)+.5,.5],cmap='RdBu_r',vmin=-3,vmax=3)
    ax[k].set_title(blk); ax[k].set_xlabel('After planned onset (s)'); ax[k].set_ylabel('event order')
    plt.colorbar(im,ax=ax[k])
plt.tight_layout(); plt.savefig('event_based.png',dpi=110)
