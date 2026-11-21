import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '24_notes_and_signals'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json,numpy as np,pandas as pd
from scipy.stats import mannwhitneyu,spearmanr
CL=['cone','cube','sphere','pyramid','cylinder']
K=json.load(open('coding.json'));R=json.load(open('../../results/16_accuracy_boosting/final_res.json'))
V=json.load(open('../../results/23_three_groups_cases_connectivity/case_res.json'))['person'];Z=json.load(open('../../results/23_three_groups_cases_connectivity/connectivity_res.json'))['occipital_second']
CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
G=lambda n:'Congenitally limited' if n in CLV else ('Typical vision then lost' if n in LVL else 'Sighted')
def acc(w,n):
    if n not in R[w]: return np.nan
    P=np.array(R[w][n]['P']);y=np.array([CL.index(q) for q in R[w][n]['y']]);return float(np.mean(P.argmax(1)==y))
rows=[]
for n in K:
    rows.append(dict(code=n,group=G(n),B1_strategy=K[n]['B1'],B2_strategy=K[n]['B2'],vividness=K[n]['vividness'],
        B1_imagery=acc('Block 1 · imagery',n),B2_imagery=acc('Block 2 · imagery',n),B2_transfer=acc('Block 2 · touch to imagery',n),B3_imagery=acc('Block 3 · imagery',n),
        pattern_closeness=V.get(n,{}).get('closeness',np.nan),occipital_alpha_touch=float(np.mean(Z['B2_touch'][n][:7])) if n in Z['B2_touch'] else np.nan,
        occipital_alpha_imagery=float(np.mean(Z['B2_img'][n][:7])) if n in Z['B2_img'] else np.nan))
T=pd.DataFrame(rows).set_index('code');T.round(3).to_csv('person_table.csv',encoding='utf-8-sig')
pd.set_option('display.width',250);print(T.drop(columns=['vividness']).round(2).to_string())
out={}
vis=lambda s:'visual' if s=='visual' else ('visual-outside' if s in ('mixed','tactile','tactile-movement','verbal-tactile') else None)
T['B2v']=T.B2_strategy.map(vis)
for col in ['pattern_closeness','B2_transfer','B2_imagery','occipital_alpha_touch','occipital_alpha_imagery']:
    for sub,mask in [('everyone',T.index==T.index),('congenital limited',T.group=='Congenitally limited'),('those who have seen (sighted+lost)',T.group!='Congenitally limited')]:
        a=T[mask&(T.B2v=='visual')][col].dropna();b=T[mask&(T.B2v=='visual-outside')][col].dropna()
        if len(a)>=2 and len(b)>=2:
            p=mannwhitneyu(a,b).pvalue;out[f'{col} | {sub}']=dict(visual=dict(n=len(a),mean=round(a.mean(),3),people=list(a.index)),visual_outside=dict(n=len(b),mean=round(b.mean(),3),people=list(b.index)),p=round(float(p),3))
            print(col,'|',sub,'| visual',len(a),round(a.mean(),3),'| non-visual',len(b),round(b.mean(),3),'| p',round(p,3))
json.dump(out,open('strategy_test.json','w'),ensure_ascii=False,indent=1)
