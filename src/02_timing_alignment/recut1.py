import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '02_timing_alignment'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, pandas as pd, json, glob, os
mne.set_log_level('ERROR')
S1=[0,12,23,33,43,50,64,71,80,90,100,112,123,135,143]
L1=['cube','cylinder','pyramid','cone','sphere','cylinder','pyramid','cone','sphere','cube','sphere','cube','cone','pyramid','cylinder']
W={'cone':1.56,'cube':1.56,'sphere':1.54,'pyramid':2.56,'cylinder':2.73}
OUT='../../data/clean_epochs/epo3'; os.makedirs(OUT,exist_ok=True); allmd=[]
for f in sorted(glob.glob('../../data/clean_continuous/*_block1_clean_raw.fif')):
    n=f.split('/')[-1].replace('_clean_raw.fif',''); t0=json.load(open(f'../../results/03_cleaning_epoching/{n}.json'))['used_onset_s']
    raw=mne.io.read_raw_fif(f,preload=True); sf=raw.info['sfreq']
    R=[(k,l,t0+s+W[l]) for k,(s,l) in enumerate(zip(S1,L1)) if t0+s+W[l]+4.0<raw.times[-1]]
    ev=np.array([[int(round(w*sf)),0,i+1] for i,(k,l,w) in enumerate(R)])
    md=pd.DataFrame({'participant':n.split('_')[0],'file':n,'block':'block1','phase':'word_and_imagery','trial':[k+1 for k,_,_ in R],'shape':[l for _,l,_ in R],
        'sound_onset_s':[round(t0+S1[k],3) for k,_,_ in R],'word_onset_s':[round(w,3) for _,_,w in R],'word_delay_s':[W[l] for _,l,_ in R],
        'time_confidence':'verified – auditory response to the word (t=3.5)'})
    ep=mne.Epochs(raw,ev,tmin=-0.5,tmax=4.0,baseline=None,preload=True,metadata=md,event_repeated='drop')
    ptp=np.ptp(ep.get_data(),axis=2).max(1)*1e6; ep.metadata['max_amplitude_uV']=np.round(ptp,1); ep.metadata['quality']=np.where(ptp>200,'high amplitude – control et','good')
    ep.save(f'{OUT}/{n}_word_aligned-epo.fif',overwrite=True,fmt='single'); allmd.append(ep.metadata)
M=pd.concat(allmd,ignore_index=True); M.to_pickle('../../results/03_cleaning_epoching/block1_v2_meta.pkl'); print(len(M),'segment', M.quality.value_counts().to_dict())
import openpyxl
from openpyxl.styles import Font
wb=openpyxl.load_workbook('../../results/03_cleaning_epoching/EEG_shape_labels.xlsx')
t='Block1 word-aligned'
if t in wb.sheetnames: del wb[t]
ws=wb.create_sheet(t); ws.append(list(M.columns))
for r in M.itertuples(index=False): ws.append(list(r))
for c in ws[1]: c.font=Font(bold=True)
ws.freeze_panes='A2'
for col in ws.columns: ws.column_dimensions[col[0].column_letter].width=max(10,min(45,max(len(str(c.value or '')) for c in col[:200])+2))
wn=wb['Method and notes']
wn.append(['Block 1 sound files',"Each .wav file contains first the same 1000 Hz beep (0–1.1 s), then the shape's name. Word onset: cone 1.56; cube 1.56; sphere 1.54; pyramid 2.56; cylinder 2.73 s (measured from the Documents\\Audacity files). Aligned to the actual word moment, a clear auditory response in 21 participants (P2–N1, t=3.5); no response at the planned beep moment. Block 1 segments are in the block1_word_aligned folder: −0.5 to +4.0 s from word onset."])
wn.append(['Note: sounds_exp2.psyexp','In the file most sounds have a duration set to 1.0 s; this should have cut off the word. Because the EEG responds to the word, the words were heard in the experiment, i.e. the version that was run differs from this setting.'])
os.makedirs('export',exist_ok=True); wb.save('export/EEG_shape_labels.xlsx')
os.makedirs('export/clean_epochs/block1_word_aligned',exist_ok=True)
os.system(f'cp {OUT}/*.fif export/clean_epochs/block1_word_aligned/')
fs=sorted(os.listdir('export/clean_epochs/block1_word_aligned'))
items=[{'stagedPath':'export/clean_epochs/block1_word_aligned/'+x,'devicePath':'<local data folder>\\EEG\\clean_epochs\\block1_word_aligned\\'+x} for x in fs]
items.append({'stagedPath':'export/EEG_shape_labels.xlsx','devicePath':'<local data folder>\\EEG\\clean_epochs\\EEG_shape_labels.xlsx'})
json.dump(items,open('commit_b1.json','w'),ensure_ascii=False); print(len(items))
