import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '24_notes_and_signals'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json,numpy as np,pandas as pd,pickle,mne,warnings;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
from scipy.stats import spearmanr
from mne.stats import permutation_cluster_test
CL=['cone','cube','sphere','pyramid','cylinder'];R=json.load(open('../../results/16_accuracy_boosting/final_res.json'))
Z=json.load(open('../../results/23_three_groups_cases_connectivity/connectivity_res.json'))['occipital_second']
CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
k=pd.read_excel('../../participants/participant_information.xlsx');G0=dict(zip(k['code'],k['analysis_group']))
out={}
rng=np.random.default_rng(0)
for excl in [[],['P19'],['P20'],['P19','P20']]:
    for c in ['B2_touch','B2_img']:
        ns=[n for n in Z[c] if (n in CLV or n in LVL or G0.get(n)=='Sighted') and n not in excl]
        o=np.array([0 if n in CLV else (1 if n in LVL else 2) for n in ns]);v=np.array([np.mean(Z[c][n][:7]) for n in ns])
        r=spearmanr(o,v).correlation;nl=[spearmanr(rng.permutation(o),v).correlation for _ in range(10000)]
        key=f"occipital ordered trend | {c} | excl: {','.join(excl) or 'none'}";out[key]=dict(rho=round(float(r),3),p=round(float((1+np.sum(np.array(nl)>=r))/10001),4));print(key,out[key])
# cluster test excluding P19
TF=pickle.load(open('../../results/20_time_frequency/tf.pkl','rb'));info=mne.read_epochs('../../data/clean_epochs/epo2/P15_block2_imagery_motionless-epo.fif',preload=False).info;adj,names=mne.channels.find_ch_adjacency(info,'eeg')
def rel(n):x=np.log(np.nanmean(TF[n]['B2_touch']['P'].reshape(-1,64,4),0))[:,1];return x-x.mean()
A=np.array([rel(n) for n in TF if n in CLV]);B=np.array([rel(n) for n in TF if G0.get(n)=='Sighted' and n!='P19'])
T,cl,pv,_=permutation_cluster_test([A,B],n_permutations=5000,adjacency=adj,tail=0,seed=0,out_type='mask')
s=sorted([(float(p),[names[i] for i in np.where(m)[0]]) for m,p in zip(cl,pv)])[:1];out['occipital cluster | touch | P19 excl']=s;print('cluster P19 excl',s)
# trial level: reported non-recognition/non-recall
for n,shapes,w in [('P10',['pyramid'],'Block 1 · imagery'),('P10',['pyramid'],'Block 2 · imagery'),('P15',['cylinder','pyramid'],'Block 1 · imagery'),('P15',['cylinder','pyramid'],'Block 2 · imagery')]:
    P=np.array(R[w][n]['P']);y=np.array(R[w][n]['y']);pr=np.array(CL)[P.argmax(1)]
    d={sh:dict(correct=int(np.sum((y==sh)&(pr==sh))),trial=int(np.sum(y==sh)),predictions=list(pr[y==sh]),correct_shape_probability=round(float(P[y==sh][:,CL.index(sh)].mean()),3)) for sh in shapes}
    d['other_shapes_accuracy']=round(float(np.mean(pr[~np.isin(y,shapes)]==y[~np.isin(y,shapes)])),3)
    out[f'trial | {n} | {w}']=d;print(n,w,d)
json.dump(out,open('robustness_res.json','w'),ensure_ascii=False,indent=1)
