import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '26_sound_and_ranking'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Brain response to sound: beep and word onset (Block 1), N1/P2 and occipital response, by group
import mne,numpy as np,glob,json,pickle,warnings,pandas as pd;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
from scipy.stats import kruskal,mannwhitneyu,spearmanr
CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
k=pd.read_excel('../../participants/participant_information.xlsx');G0=dict(zip(k['code'],k['analysis_group']))
G=lambda n:'CLV' if n in CLV else ('LVL' if n in LVL else 'Sighted')
FC='Fz FCz Cz FC1 FC2 C1 C2'.split();OC='O1 Oz O2 PO3 POz PO4'.split()
ERP={};rows={}
for f in sorted(glob.glob('../../data/clean_epochs/epo3/*-epo.fif')):
    n=f.split('/')[-1].split('_')[0];md=mne.read_epochs(f,preload=False).metadata
    raw=mne.io.read_raw_fif(f'../../data/clean_continuous/{n}_block1_clean_raw.fif',preload=True).set_eeg_reference('average').filter(1,30)
    sf=raw.info['sfreq']
    for ev,col in [('beep','sound_onset_s'),('word','word_onset_s')]:
        on=np.array(md[col].dropna(),float);events=np.c_[(on*sf).astype(int),np.zeros(len(on),int),np.ones(len(on),int)]
        ep=mne.Epochs(raw,events,tmin=-0.2,tmax=0.6,baseline=(-0.2,0),reject=dict(eeg=150e-6),preload=True)
        if len(ep)<5: continue
        e=ep.average();ERP.setdefault(ev,{})[n]=e;t=e.times;X=e.copy().pick(FC).data.mean(0)*1e6;O=e.copy().pick(OC).data.mean(0)*1e6
        w1=(t>=0.07)&(t<=0.16);w2=(t>=0.15)&(t<=0.28);wo=(t>=0.1)&(t<=0.3)
        i1=np.argmin(X[w1]);i2=np.argmax(X[w2])
        rows.setdefault(n,{}).update({f'{ev} N1 amplitude (µV)':float(X[w1][i1]),f'{ev} N1 latency (ms)':float(t[w1][i1]*1000),f'{ev} P2 amplitude (µV)':float(X[w2][i2]),f'{ev} P2 latency (ms)':float(t[w2][i2]*1000),
            f'{ev} occipital response 100–300 ms (µV)':float(O[wo].mean()),f'{ev} occipital peak absolute (µV)':float(np.abs(O[wo]).max()),f'{ev} trial':len(ep)})
    print(n,G(n),{k_:round(v,1) for k_,v in rows[n].items() if 'beep' in k_},flush=True)
T=pd.DataFrame(rows).T;T['group']=[G(n) for n in T.index];T.to_csv('sound_person.csv',encoding='utf-8-sig')
out={}
o=T.group.map({'CLV':0,'LVL':1,'Sighted':2}).values
for c in [c for c in T.columns if c!='group' and 'trial' not in c]:
    v=T[c].astype(float).values;g={gg:v[T.group.values==gg] for gg in ['CLV','LVL','Sighted']}
    r=spearmanr(o,v).correlation
    out[c]=dict(mean={gg:round(float(x.mean()),2) for gg,x in g.items()},p_CLV_Sighted=round(float(mannwhitneyu(g['CLV'],g['Sighted']).pvalue),4),p_tip_group=round(float(kruskal(*g.values()).pvalue),4),ordered_rho=round(float(r),3))
    print(c,out[c])
json.dump(out,open('sound_res.json','w'),ensure_ascii=False,indent=1)
pickle.dump(ERP,open('erp.pkl','wb'))
