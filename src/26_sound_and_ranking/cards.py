import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '26_sound_and_ranking'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import pandas as pd,numpy as np,pickle,json
from scipy.stats import mannwhitneyu,spearmanr
D=pickle.load(open('measure.pkl','rb'));G=pd.Series(D['G']);V=pd.read_csv('measure_values.csv',index_col=0);Rk=pd.read_csv('ranks.csv',index_col=0);N=V.notna().sum()
sig=[c for c in V.columns if not c.startswith(('Person','Recording'))]
# group separation per measure
separation={}
o=G.map({'Congenitally limited':0,'Typical vision then lost':1,'Sighted':2})
for c in sig:
    a=V[c][G=='Congenitally limited'].dropna();b=V[c][G=='Sighted'].dropna();m=V[c].notna()
    if len(a)>=3 and len(b)>=3: separation[c]=dict(p_CLV_Sighted=round(float(mannwhitneyu(a,b).pvalue),4),ordered_rho=round(float(spearmanr(o[m],V[c][m]).correlation),3),mean_rank={g:round(float(Rk[c][G==g].mean()),1) for g in ['Congenitally limited','Typical vision then lost','Sighted']})
json.dump(separation,open('group_separation.json','w'),ensure_ascii=False,indent=1)
for c,v in sorted(separation.items(),key=lambda x:x[1]['p_CLV_Sighted'])[:10]: print(c,v)
# profile cards
L=[]
for n in V.index:
    r=Rk.loc[n,sig].dropna();pc=(1-(r-1)/(N[sig][r.index]-1))
    top=pc.sort_values(ascending=False).head(5);bot=pc.sort_values().head(4)
    L.append(f"**{n}** ({G[n]}; years with typical vision {int(V.loc[n,'Person · years with typical vision'])}):\n- **Top ranks:** "+'; '.join(f"{c} ({int(r[c])}/{int(N[c])})" for c in top.index)+'\n- **Bottom ranks:** '+'; '.join(f"{c} ({int(r[c])}/{int(N[c])})" for c in bot.index))
open('cards.md','w').write('\n\n'.join(L));print(L[0])
