import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '22_visual_experience_grouping'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Regrouping by visual experience: occipital alpha cluster and group accuracies per window
import numpy as np,json,pickle,warnings,mne,pandas as pd;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
from mne.stats import permutation_cluster_test
from scipy.stats import kruskal
CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
k=pd.read_excel('../../participants/participant_information.xlsx');G0=dict(zip(k['code'],k['analysis_group']))
def G(n,P20=True):
    if n in CLV: return 'Congenitally limited vision'
    if n in LVL and (P20 or n!='P20'): return 'Typical vision then lost'
    return 'Sighted' if G0.get(n)=='Sighted' else None
out={}
TF=pickle.load(open('../../results/20_time_frequency/tf.pkl','rb'));info=mne.read_epochs('../../data/clean_epochs/epo2/P15_block2_imagery_motionless-epo.fif',preload=False).info;adj,names=mne.channels.find_ch_adjacency(info,'eeg')
def rel(n,c):x=np.log(np.nanmean(TF[n][c]['P'].reshape(-1,64,4),0))[:,1];return x-x.mean()
for c in ['B2_touch','B2_img']:
    for g in ['Congenitally limited vision','Typical vision then lost']:
        A=np.array([rel(n,c) for n in TF if G(n)==g]);B=np.array([rel(n,c) for n in TF if G(n)=='Sighted'])
        T,cl,pv,_=permutation_cluster_test([A,B],n_permutations=5000,adjacency=adj,tail=0,seed=0,out_type='mask')
        sig=sorted([(float(p),[names[i] for i in np.where(m)[0]],float(np.mean(A[:,m])-np.mean(B[:,m]))) for m,p in zip(cl,pv)])[:2]
        out[f'occipital alpha | {c} | {g} vs Sighted (n={len(A)} vs {len(B)})']=[dict(p=round(p,4),channels=ch,diff=round(d,3)) for p,ch,d in sig];print(c,g,len(A),[(round(p,3),ch[:8],round(d,2)) for p,ch,d in sig])
# group accuracies (final system, calibrated probabilities)
CL=['cone','cube','sphere','pyramid','cylinder'];R=json.load(open('../../results/16_accuracy_boosting/final_res.json'))
rng=np.random.default_rng(0);perms=[rng.permutation(15) for _ in range(2000)]
def acc(w,n,pi=None):
    P=np.array(R[w][n]['P']);y=np.array([CL.index(q) for q in R[w][n]['y']]);pos=np.array(R[w][n]['pos'])
    if pi is not None:
        lab=np.full(15,-1);lab[pos-1]=y;q=lab[pi];q=q[q>=0];y=q if len(q)==len(y) else rng.permutation(y)
    return np.mean(P.argmax(1)==y)
for P20 in [True,False]:
    for w in ['Block 1 · imagery','Block 2 · imagery','Block 2 · touch to imagery','Block 2 · touch']:
        grp={}
        for n in R[w]:
            g=G(n,P20)
            if g: grp.setdefault(g,[]).append(n)
        r={}
        for g,v in grp.items():
            o=np.mean([acc(w,n) for n in v]);nl=[np.mean([acc(w,n,pi) for n in v]) for pi in perms]
            r[g]=dict(n=len(v),accuracy=round(float(o),3),p_chance=float((1+np.sum(np.array(nl)>=o))/2001))
        a=np.array([acc(w,n) for g in grp for n in grp[g]]);lab=np.array([g for g in grp for n in grp[g]])
        H=lambda l:kruskal(*[a[l==g] for g in np.unique(l)]).statistic;h0=H(lab);hn=[H(rng.permutation(lab)) for _ in range(2000)]
        r['group_diff_p']=float((1+np.sum(np.array(hn)>=h0))/2001)
        out[f"accuracy | {w} | P20 {'incl' if P20 else 'excl'}"]=r;print(w,'P20' if P20 else 'P20 excl',r)
json.dump(out,open('regroup_res.json','w'),ensure_ascii=False,indent=1)
