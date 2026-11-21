import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '21_visual_experience_and_regions'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Data-driven activity comparison: group × channel × band, permutation over channel clusters (MNE)
import numpy as np,json,pickle,warnings,mne,pandas as pd;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
from mne.stats import permutation_cluster_test,permutation_cluster_1samp_test
TF=pickle.load(open('../../results/20_time_frequency/tf.pkl','rb'));BANDS=['theta','alpha','beta','gamma']
k=pd.read_excel('../../participants/participant_information.xlsx');GR=dict(zip(k['code'],k['analysis_group']))
info=mne.read_epochs('../../data/clean_epochs/epo2/P15_block2_imagery_motionless-epo.fif',preload=False).info;adj,names=mne.channels.find_ch_adjacency(info,'eeg')
def lp(n,c):P=TF[n][c]['P'];return np.log(np.nanmean(P.reshape(-1,64,4),0))   # 64×4
out={}
for tag,f in [('imagery − touch',lambda n:lp(n,'B2_img')-lp(n,'B2_touch')),('imagery (relative topography)',lambda n:(lambda x:x-x.mean(0))(lp(n,'B2_img'))),('touch (relative topography)',lambda n:(lambda x:x-x.mean(0))(lp(n,'B2_touch')))]:
    for g1,g2 in [('Congenitally low vision','Sighted'),('Later vision loss','Sighted')]:
        A=np.array([f(n) for n in TF if GR.get(n)==g1]);B=np.array([f(n) for n in TF if GR.get(n)==g2])
        for b in range(4):
            T,cl,pv,_=permutation_cluster_test([A[:,:,b],B[:,:,b]],n_permutations=5000,adjacency=adj,threshold=None,tail=0,seed=0,stat_fun=None,out_type='mask')
            # direction: mean difference
            sig=[(float(p),[names[i] for i in np.where(c)[0]],float(np.mean(A[:,c,b])-np.mean(B[:,c,b]))) for c,p in zip(cl,pv)]
            sig=sorted(sig)[:3];out[f'{tag} | {g1} vs {g2} | {BANDS[b]}']=[dict(p=round(p,4),channels=ch,diff=round(d,3)) for p,ch,d in sig]
            if sig and sig[0][0]<0.1: print(tag,'|',g1,'vs',g2,'|',BANDS[b],[(round(p,3),len(ch),ch[:10],round(d,2)) for p,ch,d in sig if p<0.2],flush=True)
# in each group imagery − touch (one-sample)
for g in ['Sighted','Congenitally low vision','Later vision loss']:
    A=np.array([lp(n,'B2_img')-lp(n,'B2_touch') for n in TF if GR.get(n)==g])
    for b in range(4):
        T,cl,pv,_=permutation_cluster_1samp_test(A[:,:,b],n_permutations=5000,adjacency=adj,tail=0,seed=0,out_type='mask')
        sig=sorted([(float(p),[names[i] for i in np.where(c)[0]],float(np.mean(A[:,c,b]))) for c,p in zip(cl,pv)])[:3]
        out[f'imagery − touch | {g} | {BANDS[b]}']=[dict(p=round(p,4),channels=ch,diff=round(d,3)) for p,ch,d in sig]
        print('single',g,BANDS[b],[(round(p,3),len(ch),round(d,2)) for p,ch,d in sig if p<0.05])
json.dump(out,open('activity_res.json','w'),ensure_ascii=False,indent=1)
