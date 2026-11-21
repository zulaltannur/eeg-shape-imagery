import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '02_timing_alignment'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, pandas as pd, os
mne.set_log_level('ERROR')
df=pd.read_csv('pertrial.csv'); OUT='../../data/clean_epochs/epo2'; os.makedirs(OUT,exist_ok=True)
TOUCH=8.0; IMAG=7.0; allmd=[]
for n,g in df.groupby('file'):
    raw=mne.io.read_raw_fif(f'../../data/clean_continuous/{n}_clean_raw.fif',preload=True); sf=raw.info['sfreq']; T=raw.times[-1]
    for ph in ('touch','imagery'):
        rows=[]
        for _,r in g.iterrows():
            if ph=='touch':
                st=r.taking_s+1.0; end=st+TOUCH; ok=end<=r.back_giving_s-0.5
                note='' if ok else 'window close to the give-back movement'
            else:
                st=r.plan_imagine_beep_s+4.0; end=st+IMAG; ok=True; note=''
            if end>T: continue
            rows.append((st,r,note))
        if not rows: continue
        ev=np.array([[int(round(st*sf)),0,i+1] for i,(st,r,nt) in enumerate(rows)])
        md=pd.DataFrame({'participant':[r.participant for _,r,_ in rows],'file':n,'block':'block2','phase':ph,'trial':[int(r.trial) for _,r,_ in rows],'shape':[r['shape'] for _,r,_ in rows],
            'window_onset_s':[round(st,2) for st,_,_ in rows],'window_s':TOUCH if ph=='touch' else IMAG,
            'taking_source':[r.taking_source for _,r,_ in rows],'back_source':[r.back_source for _,r,_ in rows],'note':[nt for _,_,nt in rows]})
        ep=mne.Epochs(raw,ev,tmin=0,tmax=(TOUCH if ph=='touch' else IMAG)-1/sf,baseline=None,preload=True,metadata=md,event_repeated='drop')
        ptp=np.ptp(ep.get_data(),axis=2).max(1)*1e6; ep.metadata['max_amplitude_uV']=np.round(ptp,1); ep.metadata['quality']=np.where(ptp>200,'high amplitude – control et','good')
        ep.save(f'{OUT}/{n}_{ph}_motionless-epo.fif',overwrite=True,fmt='single'); allmd.append(ep.metadata)
M=pd.concat(allmd,ignore_index=True); M.to_pickle('../../results/03_cleaning_epoching/block2_v2_meta.pkl')
print(M.groupby('phase').size()); print(M.quality.value_counts()); print((M['note']!='').sum(),'window warning')
# compare amplitude quality vs old epochs
old=pd.read_pickle('../../results/03_cleaning_epoching/all_meta.pkl'); old=old[old.block=='block2']
print('old block2 "check" ratio',round((old.quality!='good').mean(),3),'new',round((M.quality!='good').mean(),3))
