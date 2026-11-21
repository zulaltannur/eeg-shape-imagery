import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '27_sound_detailed'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Sound responses at single-trial level: B1 beep, B1 word, B2 "touch" beep, B2 "imagine" beep
import json as j
ACOUSTIC=j.load(open('../../results/28_word_sound_meaning_association/acoustic.json'))['feature'];INTERVAL={c:ACOUSTIC[c]['beep_word_interval'] for c in ACOUSTIC}
def word_onsets(md):
    import numpy as _np;return md['sound_onset_s'].values+_np.array([INTERVAL[s] for s in md['shape']])
import mne,numpy as np,glob,json,pickle,os,warnings,pandas as pd;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
FC='Fz FCz Cz FC1 FC2 C1 C2'.split()
PT=pd.read_csv('../../results/02_timing_alignment/pertrial.csv')
rows=[]
def trials(raw,on,meta,kind,n):
    sf=raw.info['sfreq'];ev=np.c_[(np.array(on)*sf).astype(int),np.zeros(len(on),int),np.arange(len(on))+1]
    ep=mne.Epochs(raw,ev,tmin=-0.2,tmax=0.5,baseline=(-0.2,0),preload=True,reject=None);X=ep.copy().pick(FC).get_data().mean(1)*1e6;t=ep.times
    ptp=np.ptp(ep.get_data()*1e6,axis=2).max(1)
    for i in range(len(X)):
        if ptp[i]>150: continue
        rows.append(dict(person=n,sound=kind,rank=i+1,shape=meta[i]['shape'],trial=meta[i]['trial'],N1=float(X[i][(t>=.08)&(t<=.15)].mean()),P2=float(X[i][(t>=.15)&(t<=.28)].mean())))
for f in sorted(glob.glob('../../data/clean_epochs/epo3/*-epo.fif')):
    n=f.split('/')[-1].split('_')[0];md=mne.read_epochs(f,preload=False).metadata.reset_index(drop=True)
    raw=mne.io.read_raw_fif(f'../../data/clean_continuous/{n}_block1_clean_raw.fif',preload=True).set_eeg_reference('average').filter(1,30)
    meta=[dict(shape=r['shape'],trial=int(r['trial'])) for _,r in md.iterrows()]
    trials(raw,md['sound_onset_s'].values,meta,'B1 beep',n);trials(raw,word_onsets(md),meta,'B1 word',n)
for file,g in PT.groupby('file'):
    n=file.split('_')[0];fp=f'../../data/clean_continuous/{file}_clean_raw.fif'
    if not os.path.exists(fp) or 'nocube' in file or 'third' in file: continue
    raw=mne.io.read_raw_fif(fp,preload=True).set_eeg_reference('average').filter(1,30);g=g.sort_values('trial')
    meta=[dict(shape=r['shape'],trial=int(r['trial'])) for _,r in g.iterrows()]
    trials(raw,g['plan_touch_beep_s'].values,meta,'B2 touch beep',n);trials(raw,g['plan_imagine_beep_s'].values,meta,'B2 imagery beep',n)
    print(file,flush=True)
D=pd.DataFrame(rows);D.to_csv('sound_trial_v2.csv',index=False,encoding='utf-8-sig');print(D.groupby('sound').size())
