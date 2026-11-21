import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '27_sound_detailed'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne,numpy as np,pickle,json,warnings;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
from mne.stats import spatio_temporal_cluster_test
ERP=pickle.load(open('erp_v2.pkl','rb'));CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
out={}
for ev in ['beep','word']:
    e0=next(iter(ERP[ev].values()));adj,_=mne.channels.find_ch_adjacency(e0.info,'eeg');t=e0.times;w=(t>=0)&(t<=0.5)
    A=np.array([ERP[ev][n].data[:,w].T for n in ERP[ev] if n in CLV]);B=np.array([ERP[ev][n].data[:,w].T for n in ERP[ev] if n not in CLV|LVL])
    T,cl,pv,_=spatio_temporal_cluster_test([A,B],adjacency=adj,n_permutations=5000,tail=0,seed=0,out_type='indices')
    res=[]
    for c,p in sorted(zip(cl,pv),key=lambda x:x[1])[:3]:
        ti,ci=c;tt=t[w][ti]
        res.append(dict(p=round(float(p),4),time_ms=[int(tt.min()*1000),int(tt.max()*1000)],channels=sorted(set(e0.ch_names[i] for i in ci)),diff_uV=round(float((A.mean(0)-B.mean(0))[ti,ci].mean()*1e6),2)))
    out[ev]=res;print(ev,len(A),len(B),res)
json.dump(out,open('sound_cluster_res_v2.json','w'),ensure_ascii=False,indent=1)
