import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '28_word_sound_meaning_association'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Acoustic features from the sound files (beep and word separately)
import numpy as np,soundfile as sf,librosa,json,itertools
CL=['cone','cube','sphere','pyramid','cylinder'];P=list(itertools.combinations(range(5),2))
F={};ENV={};SPEC={};MF={}
for c in CL:
    x,sr=sf.read(f'../../experiment/sounds/{c}.wav');x=x if x.ndim==1 else x.mean(1)
    w=int(0.01*sr);e=np.convolve(np.abs(x),np.ones(w)/w,'same')
    beep=e[:int(1.2*sr)];b_on=np.argmax(beep>0.1*beep.max())/sr;b_off=(len(beep)-np.argmax(beep[::-1]>0.1*beep.max()))/sr
    rest=e[int(1.2*sr):];th=0.1*rest.max();idx=np.where(rest>th)[0];w_on=1.2+idx[0]/sr;w_off=1.2+idx[-1]/sr
    word=x[int(w_on*sr):int(w_off*sr)]
    f0,vf,_=librosa.pyin(word,fmin=70,fmax=400,sr=sr);f0=f0[~np.isnan(f0)]
    S=np.abs(librosa.stft(word,n_fft=1024,hop_length=240));cent=librosa.feature.spectral_centroid(S=S,sr=sr).mean()
    mf=librosa.feature.mfcc(y=word,sr=sr,n_mfcc=13);
    F[c]=dict(beep_onset=round(b_on,3),beep_last=round(b_off,3),beep_rms=float(np.sqrt(np.mean(x[int(b_on*sr):int(b_off*sr)]**2))),word_onset=round(w_on,3),word_end=round(w_off,3),word_duration=round(w_off-w_on,3),beep_word_interval=round(w_on-b_on,3),
              word_rms=float(np.sqrt(np.mean(word**2))),pitch_hz=float(np.median(f0)) if len(f0) else None,spectral_central_hz=float(cent),
              first_100ms_rms=float(np.sqrt(np.mean(word[:int(0.1*sr)]**2))))
    ENV[c]=np.convolve(np.abs(word),np.ones(int(0.02*sr))/int(0.02*sr),'same')[::int(0.005*sr)];MF[c]=mf.mean(1);SPEC[c]=librosa.amplitude_to_db(S).mean(1)
    print(c,F[c],flush=True)
L=max(len(v) for v in ENV.values());ENV_NORM={c:np.pad(v,(0,L-len(v)))/v.max() for c,v in ENV.items()}
A={}
A['acoustic · word duration']=[abs(F[CL[i]]['word_duration']-F[CL[j]]['word_duration']) for i,j in P]
A['acoustic · sound envelope (first 400 ms)']=[1-np.corrcoef(ENV_NORM[CL[i]][:80],ENV_NORM[CL[j]][:80])[0,1] for i,j in P]
A['acoustic · spectrum (MFCC)']=[float(np.linalg.norm(MF[CL[i]]-MF[CL[j]])) for i,j in P]
A['acoustic · pitch and loudness']=[float(np.linalg.norm(np.array([(F[CL[i]]['pitch_hz']-F[CL[j]]['pitch_hz'])/30,(F[CL[i]]['word_rms']-F[CL[j]]['word_rms'])/np.std([F[c]['word_rms'] for c in CL])]))) for i,j in P]
A['acoustic · beep (control)']=[abs(F[CL[i]]['beep_rms']-F[CL[j]]['beep_rms']) for i,j in P]
json.dump(dict(feature=F,models={k:[float(v) for v in vs] for k,vs in A.items()}),open('acoustic.json','w'),ensure_ascii=False,indent=1)
