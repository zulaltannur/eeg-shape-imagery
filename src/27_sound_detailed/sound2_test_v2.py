import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '27_sound_detailed'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import pandas as pd,numpy as np,json
from scipy.stats import friedmanchisquare,wilcoxon,spearmanr,mannwhitneyu,kruskal
D=pd.read_csv('sound_trial_v2.csv');CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
D['group']=D.person.map(lambda n:'CLV' if n in CLV else ('LVL' if n in LVL else 'Sighted'))
out={}
# A) across blocks/sound types (person means, same people)
P=D.groupby(['person','sound'])[['N1','P2']].mean().unstack('sound')
for m in ['N1','P2']:
    X=P[m].dropna();out[f'A | {m} | between sound types']=dict(mean={c:round(float(X[c].mean()),2) for c in X},n=len(X),friedman_p=round(float(friedmanchisquare(*[X[c] for c in X]).pvalue),5))
    out[f'A | {m} | B1 beep vs B2 touch beep']=round(float(wilcoxon(X['B1 beep'],X['B2 touch beep']).pvalue),4)
    out[f'A | {m} | B2 touch beep vs B2 imagery beep']=round(float(wilcoxon(X['B2 touch beep'],X['B2 imagery beep']).pvalue),4)
    for g in ['CLV','LVL','Sighted']:
        Xg=X[[k in (CLV if g=='CLV' else LVL if g=='LVL' else set(X.index)-CLV-LVL) for k in X.index]]
        out[f'A | {m} | {g}']={c:round(float(Xg[c].mean()),2) for c in Xg}
# B) change across trials within a block (habituation): slope per person
for s in D.sound.unique():
    for m in ['N1','P2']:
        sl={k:spearmanr(g['rank'],g[m]).correlation for k,g in D[D.sound==s].groupby('person') if len(g)>5}
        v=np.array(list(sl.values()))
        first=D[(D.sound==s)&(D['rank']<=5)].groupby('person')[m].mean();last=D[(D.sound==s)&(D['rank']>=11)].groupby('person')[m].mean();c=first.index.intersection(last.index)
        out[f'B | {s} | {m}']=dict(mean_slope_rho=round(float(v.mean()),3),p_slope=round(float(wilcoxon(v).pvalue),4),first5=round(float(first[c].mean()),2),last5=round(float(last[c].mean()),2),p_first_last=round(float(wilcoxon(first[c],last[c]).pvalue),4),
            group_slope={g:round(float(np.mean([sl[k] for k in sl if D[D.person==k].group.iloc[0]==g])),3) for g in ['CLV','LVL','Sighted']},p_group=round(float(kruskal(*[[sl[k] for k in sl if D[D.person==k].group.iloc[0]==g] for g in ['CLV','LVL','Sighted']]).pvalue),4))
# C) by shape (Friedman within person)
for s in D.sound.unique():
    for m in ['N1','P2']:
        S=D[D.sound==s].groupby(['person','shape'])[m].mean().unstack().dropna()
        out[f'C | {s} | {m} | by shape']=dict(mean={c:round(float(S[c].mean()),2) for c in S},n=len(S),friedman_p=round(float(friedmanchisquare(*[S[c] for c in S]).pvalue),4))
for k,v in out.items(): print(k,v)
json.dump(out,open('sound2_res_v2.json','w'),ensure_ascii=False,indent=1)
