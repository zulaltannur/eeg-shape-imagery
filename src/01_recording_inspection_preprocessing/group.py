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
def t0of(j):
    on=[t for t,a,b in j['transitions'] if b==49]
    return on[0] if on else None
def subj_erp(name,ons,tmin=-0.3,tmax=1.5):
    j=json.load(open(f'../../data/preprocessing_proc/{name}.json')); t0=t0of(j)
    if t0 is None: return None
    raw=mne.io.read_raw_fif(f'../../data/preprocessing_proc/{name}_raw.fif',preload=True); sf=raw.info['sfreq']
    ev=[[int(round((t0+x)*sf)),0,1] for x in ons if (t0+x+tmax)<raw.times[-1]]
    ep=mne.Epochs(raw,np.array(ev),tmin=tmin,tmax=tmax,baseline=(-0.2,0),reject=dict(eeg=200e-6),preload=True,picks='eeg')
    if len(ep)<5: return None
    chs=[c for c in CH if c not in raw.info['bads']]
    return ep.copy().pick(chs).average().data.mean(0)*1e6, ep.times, len(ep), len(ev)
out={}
for blk,ons,pat in [('block1',S1,'../../data/preprocessing_proc/*_block1.json'),('block2',sorted(TOUCH+IMG),'../../data/preprocessing_proc/*_block2*.json')]:
    E=[];names=[]
    for f in sorted(glob.glob(pat)):
        n=f.split('/')[-1][:-5]
        if 'error' in json.load(open(f)): continue
        r=subj_erp(n,ons)
        if r: E.append(r[0]); t=r[1]; names.append((n,r[2],r[3]))
    E=np.array(E); out[blk]=(E,t,names)
    print(blk,len(E),'subjects')
np.save('group_erps.npy',{k:(v[0],v[1],v[2]) for k,v in out.items()},allow_pickle=True)
fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True)
for k,blk in enumerate(['block1','block2']):
    E,t,names=out[blk]; m=E.mean(0); se=E.std(0)/np.sqrt(len(E))
    ax[k].fill_between(t*1000,m-se,m+se,alpha=.25,color='#1f8a8a'); ax[k].plot(t*1000,m,color='#1f8a8a')
    ax[k].axvline(0,color='k',lw=.8); ax[k].axhline(0,color='gray',lw=.5)
    ax[k].set_title(f'{blk}: {len(E)} recordings, mean relative to the planned sound onset'); ax[k].set_ylabel('µV')
    # peaks
    w=(t>0.03)&(t<1.2); i=np.argmin(np.where(w,m,9)); print(blk,'most negative',round(t[i],3),round(m[i],2))
    for a,b in [(0.03,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.9)]:
        ww=(t>a)&(t<b); ii=np.argmin(np.where(ww,m,9)); jj=np.argmax(np.where(ww,m,-9))
        print('  ',a,b,'min',round(t[ii],3),round(m[ii],2),'max',round(t[jj],3),round(m[jj],2))
ax[1].set_xlabel('After planned sound onset (ms)')
plt.tight_layout(); plt.savefig('group_sound_response.png',dpi=120)
