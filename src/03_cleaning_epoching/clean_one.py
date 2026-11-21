import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '03_cleaning_epoching'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, json, os, sys, warnings
from mne.preprocessing import ICA
from mne_icalabel import label_components
warnings.filterwarnings('ignore'); mne.set_log_level('ERROR')
src=sys.argv[1]; name=os.path.basename(src)[:-4]; OUT='../../data'
if os.path.exists(f'{OUT}/rep/{name}.json'): sys.exit(0)
B64=mne.channels.make_standard_montage('biosemi64').ch_names
m128=mne.channels.make_standard_montage('biosemi128').get_positions()
pairs=json.load(open('../../results/01_recording_inspection_preprocessing/map64to128.json'))
MONT=mne.channels.make_dig_montage(ch_pos={B64[i]:m128['ch_pos'][b] for i,(a,b) in enumerate(pairs)},nasion=m128['nasion'],lpa=m128['lpa'],rpa=m128['rpa'],coord_frame='head')
# schedules
S1=[0,12,23,33,43,50,64,71,80,90,100,112,123,135,143]
L1=['cube','cylinder','pyramid','cone','sphere','cylinder','pyramid','cone','sphere','cube','sphere','cube','cone','pyramid','cylinder']
TOUCH=[0,29,57,84,112,139,166,195,222,251,279,306,335,363,391]; IMG=[13,42,70,97,125,152,179,208,235,264,292,319,348,376,404]
L2=['pyramid','cone','sphere','cylinder','cube','cone','sphere','cylinder','pyramid','cube','sphere','pyramid','cylinder','cube','cone']
S3=[0,16,33,51,67,85,102,119,135,153,169,187,203,221,238]
L3=['sphere','pyramid','cylinder','cube','cone','pyramid','cone','sphere','cylinder','cube','cone','cube','cylinder','pyramid','sphere']
rep={'file':name+'.bdf'}
try:
    raw=mne.io.read_raw_bdf(src,preload=True)
except Exception as e:
    rep['status']='unreadable (empty file)'; json.dump(rep,open(f'{OUT}/rep/{name}.json','w'),ensure_ascii=False); sys.exit(0)
sf0=raw.info['sfreq']; st=raw.get_data(picks=['Status'])[0].astype(np.int64)&0xFFFF
ons=list((np.nonzero((st[1:]==49)&(st[:-1]!=49))[0]+1)/sf0)
AB=[f'A{i}' for i in range(1,33)]+[f'B{i}' for i in range(1,33)]
raw.pick(AB); raw.reorder_channels(AB); raw.rename_channels(dict(zip(AB,B64))); raw.set_montage(MONT)
rep['duration_s']=round(raw.times[-1],1); rep['onset_markers_s']=[round(o,3) for o in ons]
OVR=os.environ.get('T0_OVERRIDE')
if OVR: ons=[float(OVR)]; rep['note']='No onset marker; onset estimated from the movement pattern (8.25 s, ±1 s; method verified on 18/22 marked recordings)'
if not ons:
    rep['status']='no onset marker – could not be split into shapes'; json.dump(rep,open(f'{OUT}/rep/{name}.json','w'),ensure_ascii=False); sys.exit(0)
t0=ons[-1]; rep['used_onset_s']=round(t0,3)
if len(ons)>1: rep['note']=f'{len(ons)} onset markers; the last one was used (earlier ones are interrupted attempts)'
raw.resample(256); raw.notch_filter([50,100])
# bad channels
tmp=raw.copy().filter(1,40); d=tmp.get_data()*1e6
sd=d.std(1); lsd=np.log(sd+1e-9); rz=(lsd-np.median(lsd))/(1.4826*np.median(np.abs(lsd-np.median(lsd)))+1e-9)
C=np.corrcoef(d)
pos=np.array([raw.info['chs'][i]['loc'][:3] for i in range(64)])
D=np.linalg.norm(pos[:,None]-pos[None],axis=2); np.fill_diagonal(D,np.inf)
nb=np.argsort(D,1)[:,:6]; mc=np.array([np.median(np.abs(C[i,nb[i]])) for i in range(64)])
rep['_neighbour_correlation_min']=round(float(mc.min()),2)
bad=[B64[i] for i in range(64) if sd[i]<0.5 or rz[i]>4 or rz[i]<-4 or mc[i]<0.4]
rep['noisy_channels_completed']=bad
if len(bad)>16: rep['warning']=f'{len(bad)} noisy channels – low recording quality'
raw.info['bads']=bad; raw.interpolate_bads(reset_bads=True)
raw.set_eeg_reference('average')
# ICA
fit=raw.copy().filter(1,100)
ica=ICA(n_components=25,method='infomax',fit_params=dict(extended=True),random_state=0,max_iter=800)
ica.fit(fit,decim=2)
lab=label_components(fit,ica,method='iclabel')
ex=[i for i,(l,p) in enumerate(zip(lab['labels'],lab['y_pred_proba'])) if l in ('eye blink','muscle artifact','heart beat','line noise','channel noise') and p>0.7]
rep['ICA_removed']=[f"{lab['labels'][i]} ({lab['y_pred_proba'][i]:.2f})" for i in ex]
ica.exclude=ex; raw.filter(0.1,40); ica.apply(raw)
raw.save(f'{OUT}/clean_continuous/{name}_clean_raw.fif',overwrite=True)
# epochs
blk=name.split('_')[1]; sf=raw.info['sfreq']; rows=[]
def add(phase,times,labels,tmax,conf,skip=()):
    for k,(x,l) in enumerate(zip(times,labels)):
        if l in skip: continue
        s=t0+x
        if s+tmax>raw.times[-1]: rows.append((phase,k+1,l,None,conf,'recording ended')); continue
        rows.append((phase,k+1,l,s,conf,''))
skip=('cube',) if 'nocube' in name else ()
if blk=='block1': add('sound_and_imagery',S1,L1,7.0,'planned time – sound delay not verified')
elif blk=='block2':
    add('touch',TOUCH,L2,12.5,('estimated onset (movement pattern) – ' if OVR else '')+'planned time – sound delay not verified',skip)
    add('imagery',IMG,L2,13.5,('estimated onset (movement pattern) – ' if OVR else '')+'planned time – sound delay not verified',skip)
elif blk=='block3':
    add('visual_perception',S3,L3,5.0,'verified – visual response time-locked')
    add('imagery',[x+5 for x in S3],L3,10.5,'verified – visual response time-locked')
TM={'sound_and_imagery':7.0,'touch':12.5,'imagery':13.5 if blk=='block2' else 10.5,'visual_perception':5.0}
import pandas as pd
ok=[r for r in rows if r[3] is not None]
rep['epoch_count']=len(ok); rep['skipped']=[f'{r[0]} #{r[1]} {r[2]}: {r[5]}' for r in rows if r[3] is None]
for ph in sorted(set(r[0] for r in ok)):
    R=[r for r in ok if r[0]==ph]
    ev=np.array([[int(round(r[3]*sf)),0,i+1] for i,r in enumerate(R)])
    md=pd.DataFrame({'participant':name.split('_')[0],'file':name,'block':blk,'phase':ph,'trial':[r[1] for r in R],'shape':[r[2] for r in R],'onset_s':[round(r[3],3) for r in R],'time_confidence':[r[4] for r in R]})
    ep=mne.Epochs(raw,ev,tmin=-0.5,tmax=TM[ph],baseline=None,preload=True,metadata=md,event_repeated='drop')
    ptp=np.ptp(ep.get_data(),axis=2).max(1)*1e6
    ep.metadata['max_amplitude_uV']=np.round(ptp,1); ep.metadata['quality']=np.where(ptp>200,'high amplitude – control et','good')
    ep.save(f'{OUT}/clean_epochs/epo/{name}_{ph}-epo.fif',overwrite=True,fmt='single')
rep['status']='done'
json.dump(rep,open(f'{OUT}/rep/{name}.json','w'),ensure_ascii=False,indent=1)
print(name,'done',len(ok),'epoch, bad channel',len(bad),'ICA',len(ex))
