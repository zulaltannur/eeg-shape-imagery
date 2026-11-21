import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '01_recording_inspection_preprocessing'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np
mne.set_log_level('ERROR')
B64 = mne.channels.make_standard_montage('biosemi64').ch_names
def load(f):
    raw = mne.io.read_raw_bdf(f, preload=True)
    st = raw.get_data(picks=['Status'])[0].astype(np.int64) & 0xFFFF
    sf = raw.info['sfreq']
    on = np.nonzero((st[1:]==49)&(st[:-1]!=49))[0]+1
    t0 = on[0]/sf
    raw.drop_channels(['Status'])
    raw.rename_channels(dict(zip(raw.ch_names, B64)))
    raw.set_montage('biosemi64')
    # channel quality
    sd = raw.get_data().std(axis=1)*1e6
    med = np.median(sd)
    bad = [raw.ch_names[i] for i in np.nonzero((sd>4*med)|(sd<med/10))[0]]
    raw.info['bads']=bad
    raw.set_eeg_reference('average')
    raw.filter(1,30)
    raw.resample(256)
    return raw,t0,bad,med
def erp_scan(raw,onsets,chs=('Fz','FCz','Cz','FC1','FC2','C1','C2')):
    sf=raw.info['sfreq']; X=raw.get_data(picks=[c for c in chs if c not in raw.info['bads']]).mean(0)*1e6
    res=[]
    for d in np.arange(-0.4,0.81,0.02):
        eps=[]
        for o in onsets:
            i=int(round((o+d)*sf)); seg=X[i-int(.1*sf):i+int(.4*sf)]
            eps.append(seg-seg[:int(.1*sf)].mean())
        e=np.mean(eps,0); t=np.arange(len(e))/sf-.1
        n1=e[(t>.07)&(t<.15)].min(); p2=e[(t>.15)&(t<.28)].max()
        res.append((round(d,2),round(p2-n1,2),round(n1,2)))
    return res
import json
out={}
# block 1: sounds_exp2 sound onsets
s1=[0,12,23,33,43,50,64,71,80,90,100,112,123,135,143]
raw,t0,bad,med=load('../../data/raw_bdf/P23_block1.bdf')
print('B1 t0',t0,'bad',bad,'median sd uV',round(med,1))
r=erp_scan(raw,[t0+x for x in s1]); best=max(r,key=lambda z:z[1]); print('B1 best shift',best); print([z for z in r if abs(z[0])<0.21])
# block 2: by_touch beeps
touch=[0,29,57,84,112,139,166,195,222,251,279,306,335,363,391]
img=[13,42,70,97,125,152,179,208,235,264,292,319,348,376,404]
raw2,t02,bad2,med2=load('../../data/raw_bdf/P23_block2_trial1.bdf')
print('B2 t0',t02,'bad',bad2,'median sd uV',round(med2,1),'dur',raw2.times[-1])
r=erp_scan(raw2,[t02+x for x in sorted(touch+img)]); best=max(r,key=lambda z:z[1]); print('B2 best shift',best); print([z for z in r if abs(z[0])<0.21])
# null: random onsets
rng=np.random.default_rng(0)
nulls=[]
for k in range(200):
    rr=rng.uniform(5,raw2.times[-1]-5,30)
    nulls.append(max(z[1] for z in erp_scan(raw2,rr)))
print('B2 null 95pct of best-shift amplitude',np.percentile(nulls,95))
nulls=[]
for k in range(200):
    rr=rng.uniform(5,raw.times[-1]-5,15)
    nulls.append(max(z[1] for z in erp_scan(raw,rr)))
print('B1 null 95pct',np.percentile(nulls,95))
