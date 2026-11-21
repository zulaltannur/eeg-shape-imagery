import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '01_recording_inspection_preprocessing'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
mne.set_log_level('ERROR')
B64 = mne.channels.make_standard_montage('biosemi64').ch_names
def load(f):
    raw = mne.io.read_raw_bdf(f, preload=True)
    st = raw.get_data(picks=['Status'])[0].astype(np.int64) & 0xFFFF
    sf = raw.info['sfreq']
    on = np.nonzero((st[1:]==49)&(st[:-1]!=49))[0]+1; t0=on[0]/sf
    raw.drop_channels(['Status']); raw.rename_channels(dict(zip(raw.ch_names,B64))); raw.set_montage('biosemi64')
    raw.filter(0.5,30); raw.resample(256)
    # bad channels after filtering
    sd=raw.get_data().std(1); med=np.median(sd)
    raw.info['bads']=[raw.ch_names[i] for i in np.nonzero((sd>3*med)|(sd<med/5))[0]]
    raw.set_eeg_reference('average')
    return raw,t0
def epochs(raw,times,reject=150e-6):
    sf=raw.info['sfreq']
    ev=np.array([[int(round(t*sf)),0,1] for t in times])
    ep=mne.Epochs(raw,ev,tmin=-0.2,tmax=0.6,baseline=(-0.2,0),reject=dict(eeg=reject),preload=True,event_repeated='drop',picks='eeg')
    return ep
def n1(ep,ch=('FCz','Cz','Fz','FC1','FC2')):
    chs=[c for c in ch if c not in ep.info['bads']]
    d=ep.copy().pick(chs).get_data().mean(1)*1e6  # trials x time
    t=ep.times; return d,t
def stat(raw,times,nrand=300,rng=np.random.default_rng(1)):
    ep=epochs(raw,times); d,t=n1(ep); w=(t>0.08)&(t<0.14)
    obs=d.mean(0)[w].mean()
    null=[]
    for k in range(nrand):
        rr=np.sort(rng.choice(np.arange(3,raw.times[-1]-3,0.05),len(times),replace=False))
        e2=epochs(raw,rr); d2,_=n1(e2); null.append(d2.mean(0)[w].mean())
    p=(np.sum(np.array(null)<=obs)+1)/(nrand+1)
    return ep,d,t,obs,p,len(ep),len(times)
res={}
raw1,t01=load('../../data/raw_bdf/P23_block1.bdf')
s1=[0,12,23,33,43,50,64,71,80,90,100,112,123,135,143]
raw2,t02=load('../../data/raw_bdf/P23_block2_trial1.bdf')
touch=[0,29,57,84,112,139,166,195,222,251,279,306,335,363,391]; img=[13,42,70,97,125,152,179,208,235,264,292,319,348,376,404]
print('bads1',raw1.info['bads'],'bads2',raw2.info['bads'])
fig,ax=plt.subplots(1,2,figsize=(11,4),sharey=True)
for k,(raw,t0,ons,name) in enumerate([(raw1,t01,s1,'Block 1 – sounds (15 sounds)'),(raw2,t02,sorted(touch+img),'Block 2 – touch/imagery beeps (30 beeps)')]):
    ep,d,t,obs,p,nk,nt=stat(raw,[t0+x for x in ons])
    print(name,'kept',nk,'/',nt,'N1 mean uV',round(obs,2),'p',round(p,3))
    m=d.mean(0); se=d.std(0)/np.sqrt(len(d))
    ax[k].fill_between(t*1000,m-se,m+se,alpha=.25,color='#1f8a8a'); ax[k].plot(t*1000,m,color='#1f8a8a')
    ax[k].axvline(0,color='k',lw=.8); ax[k].axhline(0,color='gray',lw=.5); ax[k].axvspan(80,140,color='orange',alpha=.15)
    ax[k].set_title(f'{name}\n{nk}/{nt} trials, N1 p={p:.3f}',fontsize=10); ax[k].set_xlabel('After sound onset (ms)')
ax[0].set_ylabel('Fronto-central mean (µV)')
plt.tight_layout(); plt.savefig('P23_sound_response.png',dpi=130)
