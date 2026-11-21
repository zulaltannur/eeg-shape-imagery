import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '27_sound_detailed'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Regions while the word is heard: which regions, in which rhythm, when do they activate; together (coupled) or independent?
import mne,numpy as np,glob,json,warnings,pandas as pd;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
from mne.time_frequency import tfr_morlet
from scipy.stats import wilcoxon,mannwhitneyu,spearmanr
CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
G=lambda n:'CLV' if n in CLV else ('LVL' if n in LVL else 'Sighted')
REG={'auditory (temporal)':'T7 T8 TP7 TP8 FT7 FT8','language (left fronto-temporal)':'F7 F5 FT7 FC5','fronto-central':'Fz FCz FC1 FC2','hand-touch (central)':'C3 C4 CP3 CP4 C1 C2','parietal':'P3 P4 P1 P2 Pz','vision (occipital)':'O1 Oz O2 PO3 POz PO4'}
BAND={'theta':(4,7),'alpha':(8,13),'beta':(14,25)}
freqs=np.arange(4,26,1.0);res={};tc={}
for f in sorted(glob.glob('../../data/clean_epochs/epo3/*-epo.fif')):
    n=f.split('/')[-1].split('_')[0];md=mne.read_epochs(f,preload=False).metadata.reset_index(drop=True)
    raw=mne.io.read_raw_fif(f'../../data/clean_continuous/{n}_block1_clean_raw.fif',preload=True).set_eeg_reference('average');sf=raw.info['sfreq']
    on=md['word_onset_s'].values;beep=md['sound_onset_s'].values
    ev=np.c_[(on*sf).astype(int),np.zeros(len(on),int),np.arange(len(on))+1]
    ep=mne.Epochs(raw,ev,tmin=-(on-beep).max()-0.8,tmax=1.6,baseline=None,preload=True)
    P=tfr_morlet(ep,freqs,n_cycles=freqs/2,return_itc=False,average=False,decim=4).data   # trial × channel × frequency × time
    t=ep.times[::4];out={}
    for i in range(len(on)):
        bl=(t>=-(on[i]-beep[i])-0.6)&(t<=-(on[i]-beep[i])-0.1)    # silent period before the beep
        for r,chs in REG.items():
            ci=[ep.ch_names.index(c) for c in chs.split()]
            for b,(lo,hi) in BAND.items():
                fi=(freqs>=lo)&(freqs<=hi);x=P[i][ci][:,fi].mean((0,1));d=10*np.log10(x/x[bl].mean())
                out.setdefault((r,b),[]).append(d)
    res[n]={k:np.array(v) for k,v in out.items()};tc['t']=t
    print(n,flush=True)
t=tc['t'];w=(t>=0.1)&(t<=1.0)
O={}
# 1) each region × rhythm: change after the word (dB), groups
for k in res[next(iter(res))]:
    v={n:float(res[n][k][:,w].mean()) for n in res};a=np.array(list(v.values()))
    O[f'change | {k[0]} | {k[1]}']=dict(everyone=round(float(a.mean()),2),p_everyone=round(float(wilcoxon(a).pvalue),4),**{g:round(float(np.mean([v[n] for n in v if G(n)==g])),2) for g in ['CLV','LVL','Sighted']},
        p_CLV_Sighted=round(float(mannwhitneyu([v[n] for n in v if G(n)=='CLV'],[v[n] for n in v if G(n)=='Sighted']).pvalue),4))
# 2) coupling: at trial level, relation between the auditory-region theta increase and the change in other regions (within person), then across people
for r in REG:
    if r=='auditory (temporal)': continue
    for b in BAND:
        v={n:spearmanr(res[n][('auditory (temporal)','theta')][:,w].mean(1),res[n][(r,b)][:,w].mean(1)).correlation for n in res}
        a=np.array(list(v.values()))
        O[f'coupling | auditory theta ↔ {r} {b}']=dict(mean_rho=round(float(a.mean()),3),p=round(float(wilcoxon(a).pvalue),4),**{g:round(float(np.mean([v[n] for n in v if G(n)==g])),3) for g in ['CLV','LVL','Sighted']},
            p_CLV_Sighted=round(float(mannwhitneyu([v[n] for n in v if G(n)=='CLV'],[v[n] for n in v if G(n)=='Sighted']).pvalue),4))
KB={}
for n in res:
    KB[n]=float(np.mean([spearmanr(res[n][('auditory (temporal)','theta')][:,w].mean(1),res[n][(r,'theta')][:,w].mean(1)).correlation for r in REG if r!='auditory (temporal)']))
o=np.array([0 if G(n)=='CLV' else (1 if G(n)=='LVL' else 2) for n in KB]);v=np.array(list(KB.values()))
rho=spearmanr(o,v).correlation;rng=np.random.default_rng(0);nl=[spearmanr(rng.permutation(o),v).correlation for _ in range(10000)]
O['summary | mean co-variation of auditory theta with the other 5 regions']=dict(**{g:round(float(v[o==i].mean()),3) for i,g in enumerate(['CLV','LVL','Sighted'])},p_CLV_Sighted=round(float(mannwhitneyu(v[o==0],v[o==2]).pvalue),4),ordered_rho=round(float(rho),3),p_ordered=round(float((1+np.sum(np.array(nl)>=rho))/10001),4),person={k:round(x,3) for k,x in KB.items()})
for k_,v_ in O.items(): print(k_,v_)
json.dump(O,open('region_network_res.json','w'),ensure_ascii=False,indent=1)
np.save('region_network_t.npy',t);import pickle;pickle.dump({n:{f'{a}|{b}':v.mean(0) for (a,b),v in d.items()} for n,d in res.items()},open('region_network_tc.pkl','wb'))
