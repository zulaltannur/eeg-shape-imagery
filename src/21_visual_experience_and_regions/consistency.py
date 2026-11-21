import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '21_visual_experience_and_regions'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Within-person consistency: across repetitions (same shape vs different shape) and overall pattern stability, by group
import numpy as np,json,pickle,warnings,pandas as pd;warnings.filterwarnings('ignore')
from scipy.stats import kruskal,mannwhitneyu
exec(open('../../src/21_visual_experience_and_regions/mantel.py').read().split('def pair')[0])
def cons(X,y):
    occ=np.zeros(len(y),int)
    for t in range(5): occ[y==t]=np.arange((y==t).sum())
    same=[];diff=[];gen=[]
    for i in range(len(y)):
        for j in range(i+1,len(y)):
            r=np.corrcoef(X[i],X[j])[0,1];gen.append(r)
            if occ[i]!=occ[j]: (same if y[i]==y[j] else diff).append(r)
    return np.mean(same)-np.mean(diff),np.mean(gen)
rng=np.random.default_rng(2);out={}
for cond,bset,tag in [('B2_img',['alpha'],'B2 imagery · alpha'),('B2_img',BANDS,'B2 imagery · all bands'),('B2_touch',BANDS,'B2 touch · all bands'),('B1_img',BANDS,'B1 imagery · all bands')]:
    if cond=='B1_img':
        import pickle as pk;OL=pk.load(open('../../data/derived/06_features/feats.pkl','rb'))
        R={n:((d['B1_img']['X']-d['B1_img']['X'].mean(0))/(d['B1_img']['X'].std(0)+1e-9),np.array([CL.index(q) for q in d['B1_img']['y']])) for n,d in OL.items() if 'B1_img' in d}
    else: R=feats(cond,bset)
    rows={n:cons(*R[n]) for n in R};res={}
    for g in ['Sighted','Congenitally low vision','Later vision loss']:
        v=[rows[n] for n in R if GR.get(n)==g]
        res[g]=dict(n=len(v),shape_specific=round(float(np.mean([a for a,_ in v])),4),overall_stability=round(float(np.mean([b for _,b in v])),4))
    a=[rows[n][1] for n in R if GR.get(n)=='Congenitally low vision'];b=[rows[n][1] for n in R if GR.get(n)=='Sighted']
    a2=[rows[n][0] for n in R if GR.get(n)=='Congenitally low vision'];b2=[rows[n][0] for n in R if GR.get(n)=='Sighted']
    res['CLV_vs_Sighted_overall_p']=float(mannwhitneyu(a,b).pvalue);res['CLV_vs_Sighted_shape_p']=float(mannwhitneyu(a2,b2).pvalue)
    res['person']={n:[round(float(x),4) for x in v] for n,v in rows.items()}
    out[tag]=res;print(tag,{k:v_ for k,v_ in res.items() if k!='person'},flush=True)
json.dump(out,open('consistency_res.json','w'),ensure_ascii=False,indent=1)
