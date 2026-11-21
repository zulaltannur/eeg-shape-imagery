import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '21_visual_experience_and_regions'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Region × band: where is the imagery pattern specific to congenitally low vision? (same group − sighted training; group-label permutation, max-statistic correction)
import numpy as np,json,pickle,warnings,sys,os,time,mne,pandas as pd;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
src=open('../../src/16_accuracy_boosting/final.py').read().split("out=json.load(open('final_res.json'))")[0];os.chdir('../../results/16_accuracy_boosting');exec(src)
k=pd.read_excel('../../participants/participant_information.xlsx');GRP=dict(zip(k['code'],k['analysis_group']))
TF=pickle.load(open('../../results/20_time_frequency/tf.pkl','rb'));BANDS=['theta','alpha','beta','gamma']
names=mne.read_epochs('../../data/clean_epochs/epo2/P15_block2_imagery_motionless-epo.fif',preload=False).ch_names
REG={'anterior left':'Fp1 AF7 AF3 F7 F5 F3 F1','anterior right':'Fp2 AF8 AF4 F8 F6 F4 F2','mid hat':'Fpz AFz Fz FCz Cz CPz Pz','central left':'FC5 FC3 FC1 C5 C3 C1 CP5 CP3 CP1',
     'central right':'FC6 FC4 FC2 C6 C4 C2 CP6 CP4 CP2','temporal left':'FT7 T7 TP7','temporal right':'FT8 T8 TP8','parietal left':'P9 P7 P5 P3 P1 PO7 PO3','parietal right':'P10 P8 P6 P4 P2 PO8 PO4','occipital (vision)':'O1 Oz O2 Iz POz'}
cond=sys.argv[1];part=int(sys.argv[2]);cells=[(r,b) for r in REG for b in BANDS][part::2]
def feats(r,b):
    ch=[names.index(c) for c in REG[r].split()];R={}
    for n,d in TF.items():
        if GRP.get(n) not in ('Congenitally low vision','Sighted') or cond not in d: continue
        M=np.nanmean(d[cond]['P'],1)[:,ch,BANDS.index(b)];ok=~np.isnan(M).any(1);X=np.log(M[ok]);X=(X-X.mean(0))/(X.std(0)+1e-9)
        R[n]=dict(X=X,y=d[cond]['y'][ok])
    return R
f=f'../../results/21_visual_experience_and_regions/bg_{cond}_{part}.json';st=json.load(open(f)) if os.path.exists(f) else {}
D0=sorted(n for n in TF if GRP.get(n)=='Congenitally low vision' and cond in TF[n]);S0=sorted(n for n in TF if GRP.get(n)=='Sighted' and cond in TF[n]);ALL=D0+S0
def stat(R,yy,cache,D,S,rng):
    d=[]
    for te in D:
        cd=[m for m in D if m!=te];a=[]
        for pool in (cd,S):
            Ps=[]
            for _ in range(4):
                tr=tuple(sorted(rng.choice(pool,4,replace=False)));key=(tr,te)
                if key not in cache: cache[key]=hier(R,R,list(tr),te,'X')
                Ps.append(cache[key])
            a.append(np.mean(np.mean(Ps,0).argmax(1)==yy[te]))
        d.append(a[0]-a[1])
    return float(np.mean(d))
T=time.time()
for r,b in cells:
    key=f'{r}|{b}'
    if key in st and len(st[key]['null'])>=100: continue
    R=feats(r,b);yy={n:np.array([CL.index(q) for q in R[n]['y']]) for n in R};cache={}
    s=st.get(key,{'obs':stat(R,yy,cache,D0,S0,np.random.default_rng(99)),'null':[]})
    while len(s['null'])<100:
        i=len(s['null']);rng=np.random.default_rng(1000+i);sh=list(rng.permutation(ALL));s['null'].append(stat(R,yy,cache,sh[:len(D0)],sh[len(D0):],rng))
        if time.time()-T>540: break
    st[key]=s;json.dump(st,open(f,'w'))
    print(key,round(s['obs'],3),len(s['null']),flush=True)
    if time.time()-T>540: break
