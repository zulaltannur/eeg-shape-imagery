import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '23_three_groups_cases_connectivity'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Touch region ↔ occipital connectivity (weighted phase lag index, wPLI; not affected by volume conduction) + second-by-second course of occipital alpha
import mne,numpy as np,glob,json,pickle,warnings,pandas as pd;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
from scipy.signal import butter,sosfiltfilt,hilbert
from scipy.stats import kruskal,mannwhitneyu
CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
k=pd.read_excel('../../participants/participant_information.xlsx');G0=dict(zip(k['code'],k['analysis_group']))
def G(n):return 'CLV' if n in CLV else ('LVL' if n in LVL else ('Sighted' if G0.get(n)=='Sighted' else None))
ROI={'touch left':'C5 C3 C1 CP5 CP3 CP1','touch right':'C6 C4 C2 CP6 CP4 CP2','occipital':'O1 Oz O2 PO3 POz PO4 PO7 PO8 Iz'}
B={'alpha':(8,13),'beta':(14,25)};SF=256
def wpli(a,b):
    c=np.imag(a*np.conj(b));return np.abs(c.mean())/(np.abs(c).mean()+1e-12)
res={}
for f in sorted(glob.glob('../../data/clean_epochs/epo2/*_block2*_touch_motionless-epo.fif')):
    n=f.split('/')[-1].split('_')[0]
    if 'nocube' in f or 'third' in f or not G(n): continue
    for cond,ff in [('touch',f),('imagery',f.replace('touch','imagery'))]:
        ep=mne.read_epochs(ff,preload=True).set_eeg_reference('average');X=ep.get_data();ch=ep.ch_names
        for bn,(lo,hi) in B.items():
            sos=butter(4,(lo,hi),btype='band',fs=SF,output='sos');A=hilbert(sosfiltfilt(sos,X,axis=2),axis=2)
            for r in ['touch left','touch right']:
                ia=[ch.index(c) for c in ROI[r].split()];ib=[ch.index(c) for c in ROI['occipital'].split()]
                v=np.mean([[wpli(A[t,i],A[t,j]) for i in ia for j in ib] for t in range(len(X))])
                res.setdefault(n,{})[f'{cond}|{bn}|{r}↔occipital']=float(v)
    print(n,G(n),{k_:round(v_,3) for k_,v_ in res[n].items() if 'alpha' in k_},flush=True)
out={'person':res,'test':{}}
for key in res[next(iter(res))]:
    g={gg:[res[n][key] for n in res if G(n)==gg] for gg in ['CLV','LVL','Sighted']}
    out['test'][key]=dict(mean={gg:round(float(np.mean(v)),4) for gg,v in g.items()},n={gg:len(v) for gg,v in g.items()},
        p_CLV_Sighted=float(mannwhitneyu(g['CLV'],g['Sighted']).pvalue),p_tip_group=float(kruskal(*g.values()).pvalue))
    print(key,out['test'][key])
# second-by-second course of occipital alpha (relative: occipital − all channels)
TF=pickle.load(open('../../results/20_time_frequency/tf.pkl','rb'));info=mne.read_epochs('../../data/clean_epochs/epo2/P15_block2_imagery_motionless-epo.fif',preload=False).info
io=[info.ch_names.index(c) for c in 'PO7 PO3 O1 Iz Oz POz PO4'.split()];zs={}
for n,d in TF.items():
    if not G(n): continue
    for c in ['B2_touch','B2_img']:
        L=np.log(np.nanmean(d[c]['P'][:,:,:,1],0));zs.setdefault(c,{})[n]=(L[:,io].mean(1)-L.mean(1)).tolist()
out['occipital_second']=zs
for c in zs:
    for s in range(7):
        a=[zs[c][n][s] for n in zs[c] if G(n)=='CLV'];b=[zs[c][n][s] for n in zs[c] if G(n)=='Sighted'];o=[zs[c][n][s] for n in zs[c] if G(n)=='LVL']
        out['test'][f'occipital alpha | {c} | {s}-{s+1} s']=dict(CLV=round(float(np.mean(a)),3),LVL=round(float(np.mean(o)),3),Sighted=round(float(np.mean(b)),3),p_CLV_Sighted=float(mannwhitneyu(a,b).pvalue))
        print(c,s,out['test'][f'occipital alpha | {c} | {s}-{s+1} s'])
json.dump(out,open('../../results/23_three_groups_cases_connectivity/connectivity_res.json','w'),ensure_ascii=False,indent=1)
