import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '13_tactile_hmm_touchsim'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne,numpy as np,pandas as pd,os,pickle,warnings;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
from scipy.signal import butter,sosfiltfilt,hilbert
SF=256;PT=pd.read_csv('../../results/02_timing_alignment/pertrial.csv')
G=dict(zip(*pd.read_excel('../../participants/participant_information.xlsx')[['code','analysis_group']].values.T))
ROI={'left_mid':['C3','CP3','C5','FC3','C1','CP1'],'right_mid':['C4','CP4','C6','FC4','C2','CP2'],'parietal':['P1','Pz','P2','P3','P4','CPz'],'occipital':['O1','Oz','O2','PO3','PO4','POz']}
BANDS={'theta':(4,7),'mu':(8,13),'beta':(14,25)}
W=int(0.5*SF);ST=int(0.25*SF);hp=butter(4,1,btype='high',fs=SF,output='sos')
D={}
for file,d in PT.groupby('file'):
    n=d.participant.iloc[0]
    if n not in G or 'nocube' in file or 'third' in file or n in D: continue
    f=f'../../data/clean_continuous/{file}_clean_raw.fif'
    if not os.path.exists(f): continue
    raw=mne.io.read_raw_fif(f,preload=True).set_eeg_reference('average');y=raw.get_data();ch=raw.ch_names
    env={}
    for b,(lo,hi) in BANDS.items():
        e=np.abs(hilbert(sosfiltfilt(butter(4,[lo,hi],btype='band',fs=SF,output='sos'),y,axis=1),axis=1))**2
        for r,cs in ROI.items(): env[(r,b)]=e[[ch.index(c) for c in cs]].mean(0)
    yh=sosfiltfilt(hp,y,axis=1)
    trials=[]
    for _,r in d.sort_values('trial').iterrows():
        if n=='P23' and r.trial==15: continue
        a,bnd=int((r.taking_s+0.5)*SF),int((r.back_giving_s-0.5)*SF)
        F=[];ok=[]
        for s in range(a,bnd-W,ST):
            F.append([np.log(env[k][s:s+W].mean()) for k in env]);ok.append(np.ptp(yh[:,s:s+W])*1e6<150)
        trials.append(dict(F=np.array(F),ok=np.array(ok),shape=r['shape'],trial=int(r.trial)))
    allF=np.vstack([t['F'][t['ok']] for t in trials]);mu=allF.mean(0);sd=allF.std(0)+1e-9
    for t in trials: t['F']=(t['F']-mu)/sd
    D[n]=dict(trials=trials,keys=[f'{a}_{b}' for a,b in env]);print(n,len(trials),'trial',sum(len(t['F']) for t in trials),'window',flush=True)
pickle.dump(D,open('hmm_feats.pkl','wb'))
