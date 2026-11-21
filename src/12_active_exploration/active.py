"""Block 2 active exploration (taking the shape → giving it back). Exploratory analysis."""
import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '12_active_exploration'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne,numpy as np,pandas as pd,glob,os,pickle,warnings,json;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
SF=256;PT=pd.read_csv('../../results/02_timing_alignment/pertrial.csv')
G=dict(zip(*pd.read_excel('../../participants/participant_information.xlsx')[['code','analysis_group']].values.T))
FB={'theta':(4,7),'alpha':(8,13),'beta':(14,25),'gamma':(26,40)}
D={}
for file,d in PT.groupby('file'):
    n=d.participant.iloc[0]
    if n not in G or 'nocube' in file or 'third' in file or n in D: continue
    f=f'../../data/clean_continuous/{file}_clean_raw.fif'
    if not os.path.exists(f): continue
    raw=mne.io.read_raw_fif(f,preload=True).set_eeg_reference('average');y=raw.get_data();ch=raw.ch_names
    seg=[];lab=[];tri=[];rep=[]
    cnt={}
    for _,r in d.sort_values('trial').iterrows():
        if n=='P23' and r.trial==15: continue
        a,b=r.taking_s+0.5,r.back_giving_s-0.5
        k=cnt.get(r['shape'],0);cnt[r['shape']]=k+1
        for s in np.arange(a,b-1,0.5):
            x=y[:,int(s*SF):int((s+1)*SF)]
            if x.shape[1]<SF or np.ptp(x)*1e6>300: continue
            p,fr=mne.time_frequency.psd_array_welch(x,SF,fmin=2,fmax=40,n_fft=SF,verbose=False)
            seg.append(np.concatenate([np.log(p[:,(fr>=lo)&(fr<=hi)].mean(1)) for lo,hi in FB.values()]));lab.append(r['shape']);tri.append(r.trial);rep.append(k)
    D[n]=dict(X=np.array(seg),y=np.array(lab),tri=np.array(tri),rep=np.array(rep),ch=ch)
    print(n,len(seg),'segment',flush=True)
pickle.dump(D,open('active_feats.pkl','wb'))
def z(X):return (X-X.mean(0))/(X.std(0)+1e-9)
# 1) within person: leave-repetition-out (in each fold one trial of each shape is left out -> balanced)
rng=np.random.default_rng(1);out={}
def within(n,perm=None):
    X=z(D[n]['X']);y=D[n]['y'].copy();tri=D[n]['tri'];rep=D[n]['rep']
    if perm is not None:  # label shuffle at trial level
        ut=np.unique(tri);mp=dict(zip(ut,[y[tri==t][0] for t in ut]));sh=dict(zip(ut,np.array([mp[t] for t in ut])[perm[:len(ut)]%len(ut)]))
        y=np.array([sh[t] for t in tri])
    hits=[]
    for k in np.unique(rep):
        te=rep==k;tr=~te
        if len(np.unique(y[tr]))<2: continue
        c=make_pipeline(StandardScaler(),LogisticRegression(C=0.01,max_iter=2000,class_weight='balanced')).fit(X[tr],y[tr])
        P=c.predict_proba(X[te]);cls=c.classes_
        for t in np.unique(tri[te]):  # decision at trial level: mean of segment probabilities
            m=tri[te]==t;hits.append(cls[P[m].mean(0).argmax()]==y[te][m][0])
    return float(np.mean(hits))
W={n:within(n) for n in D}
null=[]
for i in range(200):
    pm=rng.permutation(15);null.append(np.mean([within(n,pm) for n in D]))
mw=np.mean(list(W.values()));pw=(np.sum(np.array(null)>=mw)+1)/201
out['person_within']=dict(acc=float(mw),p=float(pw),null95=float(np.percentile(null,95)),person=W)
print(f'WITHIN-PERSON active exploration: mean accuracy {mw:.3f} (chance 95% {np.percentile(null,95):.3f}) p={pw:.3f}',flush=True)
print({k:round(v,2) for k,v in sorted(W.items(),key=lambda x:-x[1])},flush=True)
json.dump(out,open('active_res.json','w'),ensure_ascii=False)
