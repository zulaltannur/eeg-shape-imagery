import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '01_recording_inspection_preprocessing'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, json, glob, warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore'); mne.set_log_level('ERROR')
S3=[0,16,33,51,67,85,102,119,135,153,169,187,203,221,238]
SEQ=['Sphere','Pyramid','Cylinder','Cube','Cone','Pyramid','Cone','Sphere','Cylinder','Cube','Cone','Cube','Cylinder','Pyramid','Sphere']
CL=['Sphere','Pyramid','Cylinder','Cube','Cone']; y=np.array([CL.index(s) for s in SEQ])
SKIP={'P19_block3','P01_block3'}
subs={}
for f in sorted(glob.glob('../../data/preprocessing_proc/*_block3.json')):
    j=json.load(open(f)); n=j['name']
    if 'error' in j or n in SKIP: continue
    on=[t for t,a,b in j['transitions'] if b==49]
    if not on: continue
    raw=mne.io.read_raw_fif(f'../../data/preprocessing_proc/{n}_raw.fif',preload=True); raw.interpolate_bads(reset_bads=True); raw.set_eeg_reference('average')
    sf=raw.info['sfreq']; X=raw.get_data()*1e6; t0=on[0]
    P=[];I=[]
    for x in S3:
        i=int(round((t0+x)*sf)); seg=X[:,i-int(.2*sf):i+int(15*sf)]
        if seg.shape[1]<int(15.2*sf): seg=np.pad(seg,((0,0),(0,int(15.2*sf)-seg.shape[1])),mode='edge')
        seg=seg-seg[:,:int(.2*sf)].mean(1,keepdims=True)
        # perception ERP features: 50ms bins 0.05-0.8 s
        b0=int(.2*sf); P.append(np.concatenate([seg[:,b0+int(a*sf):b0+int((a+.05)*sf)].mean(1) for a in np.arange(0.05,0.8,0.05)]))
        # band power: perception (0.5-5s) and imagery (5.5-11s)
        def bp(s):
            fr=np.fft.rfftfreq(s.shape[1],1/sf); F=np.abs(np.fft.rfft(s*np.hanning(s.shape[1]),axis=1))**2
            return np.concatenate([np.log(F[:,(fr>=lo)&(fr<hi)].mean(1)) for lo,hi in [(4,8),(8,13),(13,20),(20,30)]])
        I.append(np.concatenate([bp(seg[:,b0+int(.5*sf):b0+int(5*sf)]),bp(seg[:,b0+int(5.5*sf):b0+int(11*sf)])]))
    P=np.array(P); I=np.array(I); nc=64*4
    z=lambda A:(A-A.mean(0))/(A.std(0)+1e-9)
    subs[n]={'erp':z(P),'bp_perc':z(I[:,:nc]),'bp_imag':z(I[:,nc:])}
names=list(subs); print(len(names),'participant:',names)
def loso(feat,labels):
    acc=[]
    for k,n in enumerate(names):
        Xtr=np.vstack([subs[m][feat] for m in names if m!=n]); ytr=np.concatenate([labels for m in names if m!=n])
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=0.01,max_iter=3000)).fit(Xtr,ytr)
        acc.append(clf.score(subs[n][feat],labels))
    return float(np.mean(acc)),acc
rng=np.random.default_rng(0); out={}
for feat,lab in [('erp','Perception – visual response wave (0–0.8 s)'),('bp_perc','Perception – frequency power (0.5–5 s)'),('bp_imag','Imagery – frequency power (5.5–11 s)')]:
    a,per=loso(feat,y)
    null=[loso(feat,rng.permutation(y))[0] for _ in range(200)]  # same shuffled sequence for all subjects (keeps order structure)
    p=(np.sum(np.array(null)>=a)+1)/201
    out[feat]=dict(label=lab,acc=a,per_subject=per,null_mean=float(np.mean(null)),null95=float(np.percentile(null,95)),p=p)
    print(f'{lab}: accuracy {a:.3f} | same-order shuffle mean {np.mean(null):.3f}, 95% {np.percentile(null,95):.3f} | p={p:.3f}')
json.dump(out,open('dec3_res.json','w'),ensure_ascii=False)
