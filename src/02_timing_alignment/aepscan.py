import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '02_timing_alignment'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, json, glob
mne.set_log_level('ERROR')
FC=['Fz','FCz','Cz','FC1','FC2','C1','C2']; INF=['P9','P10','TP7','TP8','PO7','PO8']
S1=[0,12,23,33,43,50,64,71,80,90,100,112,123,135,143]
TOUCH=[0,29,57,84,112,139,166,195,222,251,279,306,335,363,391]; IMG=[13,42,70,97,125,152,179,208,235,264,292,319,348,376,404]
SH=np.round(np.arange(-4,4.001,0.02),2)
def scan(pattern,ons):
    allm=[]
    for f in sorted(glob.glob(pattern)):
        n=f.split('/')[-1].replace('_clean_raw.fif','')
        if 'P19' in n: continue
        rep=json.load(open(f'../../results/03_cleaning_epoching/{n}.json')); t0=rep.get('used_onset_s')
        if t0 is None: continue
        raw=mne.io.read_raw_fif(f,preload=True).filter(1,15); sf=raw.info['sfreq']
        x=(raw.get_data(picks=FC).mean(0)-raw.get_data(picks=INF).mean(0))*1e6
        M=[]
        for s in SH:
            eps=[]
            for o in ons:
                i=int(round((t0+o+s)*sf)); 
                if i-26<0 or i+80>len(x): continue
                e=x[i-26:i+80]; e=e-e[:26].mean()
                if np.ptp(e)<80: eps.append(e)
            m=np.mean(eps,0) if eps else np.zeros(106)
            t=np.arange(106)/sf-0.1
            M.append(m[(t>=.16)&(t<.26)].mean()-m[(t>=.07)&(t<.14)].mean())  # P2-N1
        allm.append(M)
    A=np.array(allm); g=A.mean(0); se=A.std(0)/np.sqrt(len(A)); tv=g/se
    i=np.argmax(tv); return A,g,tv,i
out={}
for lab,pat,ons in [('block2','../../data/clean_continuous/*_block2*_clean_raw.fif',sorted(TOUCH+IMG)),('block1','../../data/clean_continuous/*_block1_clean_raw.fif',S1)]:
    A,g,tv,i=scan(pat,ons)
    o=np.argsort(tv)[::-1]; top=[]
    for j in o:
        if all(abs(SH[j]-SH[k])>0.3 for k in top): top.append(j)
        if len(top)==4: break
    print(lab,'recording',len(A),'strongest shifts:',[(float(SH[j]),round(float(g[j]),2),round(float(tv[j]),2)) for j in top])
    np.save(f'aepscan_{lab}.npy',np.vstack([SH,g,tv]))
