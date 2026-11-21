import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '21_visual_experience_and_regions'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,json,pickle,warnings,pandas as pd;warnings.filterwarnings('ignore')
from scipy.stats import mannwhitneyu
exec(open('../../src/21_visual_experience_and_regions/mantel.py').read().split('def feats')[0])
out={}
for cond in ['B2_img','B2_touch']:
    rows={}
    for n,d in TF.items():
        if cond not in d: continue
        P=d[cond]['P'];clean=float(np.mean(~np.isnan(P[:,:,0,0])))
        L=np.log(np.nanmean(P,1));sd=float(np.nanmean(np.nanstd(L[:,:,1],0)))   # trial-to-trial SD of alpha log power
        rows[n]=(clean,sd)
    res={}
    for g in ['Sighted','Congenitally low vision','Later vision loss']:
        v=[rows[n] for n in rows if GR.get(n)==g];res[g]=dict(n=len(v),clean_second_ratio=round(float(np.mean([a for a,_ in v])),3),alpha_trials_between_SS=round(float(np.mean([b for _,b in v])),3))
    for i,key in enumerate(['clean_second_ratio','alpha_trials_between_SS']):
        res[key+'_p']=float(mannwhitneyu([rows[n][i] for n in rows if GR.get(n)=='Congenitally low vision'],[rows[n][i] for n in rows if GR.get(n)=='Sighted']).pvalue)
    out[cond]=res;print(cond,res)
json.dump(out,open('noise_res.json','w'),ensure_ascii=False,indent=1)
