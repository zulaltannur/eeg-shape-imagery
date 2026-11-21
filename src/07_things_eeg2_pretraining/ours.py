"""Bring our recordings into THINGS format: 17 posterior channels, average reference, 100 Hz, [-0.2, 0.8] s, baseline -0.2..0."""
import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '07_things_eeg2_pretraining'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne,numpy as np,glob,pickle,warnings,json
warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
CH=json.load(open('meta.json'))['ch_names'];CLEAN='../../data'
def prep(ep):
    e=ep.copy().set_eeg_reference('average').filter(0.1,40).resample(100).pick(CH);return e.get_data(),e.times
def seg(X,t,onsets):
    out=[]
    for o in onsets:
        i=int(round((o-0.2-t[0])*100))
        if i<0 or i+100>X.shape[-1]: continue
        s=X[...,i:i+100];out.append(s-s[...,:20].mean(-1,keepdims=True))
    return np.stack(out,1) if out else None  # (trial, nseg, ch, 100)
P=__import__('pandas').read_excel('../../participants/participant_information.xlsx');G=list(P.code)
D={}
for n in G:
    d={}
    f=glob.glob(f'{CLEAN}/clean_epochs/epo3/{n}_*-epo.fif')
    if f:
        ep=mne.read_epochs(f[0],preload=True);md=ep.metadata.reset_index(drop=True);X,t=prep(ep)
        d['B1_word']=dict(S=seg(X,t,[0.0]),y=md['shape'].values,pos=md.trial.values)
        d['B1_img']=dict(S=seg(X,t,np.arange(0.5,3.3,0.5)),y=md['shape'].values,pos=md.trial.values)
    f=[x for x in glob.glob(f'{CLEAN}/clean_epochs/epo2/{n}_block2*_touch_motionless-epo.fif') if 'nocube' not in x and 'third' not in x]
    for cond,ff in ([('B2_touch',f[0]),('B2_img',f[0].replace('touch','imagery'))] if f else []):
        ep=mne.read_epochs(ff,preload=True);md=ep.metadata.reset_index(drop=True);X,t=prep(ep)
        S=seg(X,t,np.arange(t[0]+0.2,t[-1]-0.8,0.5));k=~((n=='P23')&(md.trial.values==15))
        d[cond]=dict(S=S[k],y=md['shape'].values[k],pos=md.trial.values[k])
    f=glob.glob(f'{CLEAN}/clean_epochs/epo/{n}_block3_visual_perception-epo.fif')
    if f:
        ep=mne.read_epochs(f[0],preload=True);md=ep.metadata.reset_index(drop=True);X,t=prep(ep)
        d['B3_picture']=dict(S=seg(X,t,[0.0]),y=md['shape'].values,pos=md.trial.values)
        d['B3_viewing']=dict(S=seg(X,t,np.arange(1.0,4.1,0.5)),y=md['shape'].values,pos=md.trial.values)
        ei=mne.read_epochs(f[0].replace('visual_perception','imagery'),preload=True);mi=ei.metadata.reset_index(drop=True);X,t=prep(ei)
        d['B3_img']=dict(S=seg(X,t,np.arange(1.0,8.1,0.5)),y=mi['shape'].values,pos=mi.trial.values)
    D[n]=d;print(n,{k:v['S'].shape for k,v in d.items()},flush=True)
pickle.dump(D,open('ours.pkl','wb'))
