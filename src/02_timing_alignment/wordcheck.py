import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '02_timing_alignment'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, json, glob
mne.set_log_level('ERROR')
FC=['Fz','FCz','Cz','FC1','FC2','C1','C2']; INF=['P9','P10','TP7','TP8','PO7','PO8']
S1=[0,12,23,33,43,50,64,71,80,90,100,112,123,135,143]
L1=['cube','cylinder','pyramid','cone','sphere','cylinder','pyramid','cone','sphere','cube','sphere','cube','cone','pyramid','cylinder']
W={'cone':1.56,'cube':1.56,'sphere':1.54,'pyramid':2.56,'cylinder':2.73}
def erp(evt):
    E=[]
    for f in sorted(glob.glob('../../data/clean_continuous/*_block1_clean_raw.fif')):
        n=f.split('/')[-1].replace('_clean_raw.fif','')
        if 'P19' in n: continue
        t0=json.load(open(f'../../results/03_cleaning_epoching/{n}.json'))['used_onset_s']
        raw=mne.io.read_raw_fif(f,preload=True).filter(1,15); sf=raw.info['sfreq']
        x=(raw.get_data(picks=FC).mean(0)-raw.get_data(picks=INF).mean(0))*1e6
        eps=[]
        for o in evt:
            i=int(round((t0+o)*sf)); e=x[i-26:i+128]; e=e-e[:26].mean()
            if len(e)==154 and np.ptp(e)<80: eps.append(e)
        E.append(np.mean(eps,0))
    E=np.array(E); t=np.arange(154)/256-0.1
    n1=E[:,(t>=.08)&(t<.15)].mean(1); p2=E[:,(t>=.16)&(t<.28)].mean(1); d=p2-n1
    return round(d.mean(),2), round(d.mean()/(d.std()/np.sqrt(len(d))),2), len(E)
print('beep onset (0 s)      P2–N1, t, n:',erp(S1))
print('word onset (actual) P2–N1, t, n:',erp([s+W[l] for s,l in zip(S1,L1)]))
print('beep end (1.1 s)        P2–N1, t, n:',erp([s+1.1 for s in S1]))
