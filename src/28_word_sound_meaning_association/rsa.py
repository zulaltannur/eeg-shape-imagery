import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '28_word_sound_meaning_association'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Over time after the word is heard: which model does the brain dissimilarity matrix resemble? (beep: control)
import mne,numpy as np,glob,json,itertools,warnings,pandas as pd;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
from scipy.stats import spearmanr,rankdata
CL=['cone','cube','sphere','pyramid','cylinder'];P=list(itertools.combinations(range(5),2));PERM=list(itertools.permutations(range(5)))
M={k:np.array(v) for k,v in json.load(open('models.json')).items()}
CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
def cvrdm(X,y):   # X: trial × feature; leave-repetition-out cross-validated correlation distance
    occ=np.zeros(len(y),int)
    for t in range(5): occ[y==t]=np.arange((y==t).sum())
    Ds=[]
    for k in range(occ.max()+1):
        A=[X[(y==t)&(occ==k)].mean(0) if ((y==t)&(occ==k)).any() else None for t in range(5)];B=[X[(y==t)&(occ!=k)].mean(0) if ((y==t)&(occ!=k)).any() else None for t in range(5)]
        D=np.full((5,5),np.nan)
        for i in range(5):
            for j in range(5):
                if A[i] is not None and B[j] is not None: D[i,j]=1-np.corrcoef(A[i],B[j])[0,1]
        Ds.append(D)
    D=np.nanmean(Ds,0);D=(D+D.T)/2;return np.array([D[i,j] for i,j in P])
bins=[(a,a+0.1) for a in np.arange(0.0,0.75,0.05)]
NEURAL={'word':{},'beep':{}}
for f in sorted(glob.glob('../../data/clean_epochs/epo3/*-epo.fif')):
    n=f.split('/')[-1].split('_')[0];md=mne.read_epochs(f,preload=False).metadata.reset_index(drop=True)
    raw=mne.io.read_raw_fif(f'../../data/clean_continuous/{n}_block1_clean_raw.fif',preload=True).set_eeg_reference('average').filter(1,30);sf=raw.info['sfreq']
    for ev,col in [('word','word_onset_s'),('beep','sound_onset_s')]:
        on=md[col].values;E=np.c_[(on*sf).astype(int),np.zeros(len(on),int),np.arange(len(on))+1]
        ep=mne.Epochs(raw,E,tmin=-0.2,tmax=0.85,baseline=(-0.2,0),preload=True);X=ep.get_data()*1e6;t=ep.times;y=np.array([CL.index(s) for s in md['shape']])
        ok=np.ptp(X,axis=2).max(1)<150;X=X[ok];y=y[ok]
        NEURAL[ev][n]=np.array([cvrdm(X[:,:,(t>=a)&(t<b)].mean(2),y) for a,b in bins])
    print(n,flush=True)
out={}
def grp(n):return 'CLV' if n in CLV else ('LVL' if n in LVL else 'Sighted')
for ev in NEURAL:
    subs=list(NEURAL[ev]);R=np.array([NEURAL[ev][s] for s in subs])   # person × time × 10
    for mn,m in M.items():
        def stat(mm,who=None):
            idx=[i for i,s in enumerate(subs) if who is None or grp(s)==who]
            return np.array([np.mean([spearmanr(R[i,b],mm).correlation for i in idx]) for b in range(len(bins))])
        obs=stat(m)
        perm=np.array([stat(np.array([ (lambda Q:Q[pi[i],pi[j]])(np.array(full)) for i,j in P])) for pi in PERM for full in [None]][:0]) if False else None
        # exact permutation: the model's shape labels are shuffled in 120 ways
        Mf=np.zeros((5,5));[Mf.__setitem__((i,j),v) or Mf.__setitem__((j,i),v) for (i,j),v in zip(P,m)]
        nul=np.array([stat(np.array([Mf[p[i],p[j]] for i,j in P])) for p in PERM])
        mx=nul.max(1);pz=[(np.sum(nul[:,b]>=obs[b]-1e-12))/120 for b in range(len(bins))];pc=[(np.sum(mx>=obs[b]-1e-12))/120 for b in range(len(bins))]
        best=int(np.argmax(obs))
        out[f'{ev} | {mn}']=dict(time=[f'{int(a*1000)}–{int(b*1000)}' for a,b in bins],rho=[round(float(x),3) for x in obs],p=[round(float(x),3) for x in pz],p_time_corrected=[round(float(x),3) for x in pc],
            best_window=f'{int(bins[best][0]*1000)}–{int(bins[best][1]*1000)} ms',best_rho=round(float(obs[best]),3),best_p=round(float(pz[best]),3),best_p_corr=round(float(pc[best]),3),
            groups={g:[round(float(x),3) for x in stat(m,g)] for g in ['CLV','LVL','Sighted']})
        print(ev,mn,out[f'{ev} | {mn}']['best_window'],out[f'{ev} | {mn}']['best_rho'],out[f'{ev} | {mn}']['best_p'],out[f'{ev} | {mn}']['best_p_corr'],flush=True)
json.dump(out,open('rsa_res.json','w'),ensure_ascii=False,indent=1)
import pickle;pickle.dump(NEURAL,open('neural_rdm.pkl','wb'))
