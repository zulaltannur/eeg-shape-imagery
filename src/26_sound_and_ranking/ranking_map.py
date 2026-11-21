import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '26_sound_and_ranking'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import pickle,numpy as np,pandas as pd,json,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
D=pickle.load(open('measure.pkl','rb'));M=pd.DataFrame(D['M']).T;G=pd.Series(D['G'])
order=[c for c in M.columns if c.startswith('Readability')]+[c for c in M.columns if c.startswith(('Curvature','Pointedness'))]+[c for c in M.columns if c.startswith('Pattern')]+[c for c in M.columns if c.startswith('Stability')]+\
      [c for c in M.columns if c.startswith(('Vision region','Hand-touch','Touch–occipital','To imagery'))]+[c for c in M.columns if c.startswith('Sound')]+[c for c in M.columns if c.startswith(('Person','Recording'))]
M=M[order].astype(float)
group_order={'Congenitally limited':0,'Typical vision then lost':1,'Sighted':2};rows=sorted(M.index,key=lambda n:(group_order[G[n]],n));M=M.loc[rows]
Rk=M.rank(ascending=False,method='min');N=M.notna().sum();P=1-(Rk-1)/(N-1)
Rk.to_csv('ranks.csv',encoding='utf-8-sig');M.round(4).to_csv('measure_values.csv',encoding='utf-8-sig')
fig,ax=plt.subplots(figsize=(0.42*len(order)+4,0.42*len(rows)+5.5))
im=ax.imshow(P.values,cmap='Blues',vmin=0,vmax=1,aspect='auto')
for i in range(len(rows)):
    for j in range(len(order)):
        if not np.isnan(Rk.values[i,j]): ax.text(j,i,int(Rk.values[i,j]),ha='center',va='center',fontsize=6.5,color='white' if P.values[i,j]>0.6 else '#1f2328')
ax.set_xticks(range(len(order)),order,rotation=60,ha='right',fontsize=7.5);ax.set_yticks(range(len(rows)),[f"{n}" for n in rows],fontsize=8)
b=[i for i in range(1,len(rows)) if G[rows[i]]!=G[rows[i-1]]]
for y in b: ax.axhline(y-0.5,color='#1f2328',lw=1.2)
for i,g in enumerate(['Congenitally limited vision','Typical vision then lost','Sighted']):
    idx=[k for k,n in enumerate(rows) if group_order[G[n]]==i];ax.annotate(g,xy=(-0.075,1-(np.mean(idx)+0.5)/len(rows)),xycoords='axes fraction',ha='center',va='center',fontsize=8.5,rotation=90,weight='bold')
sec=[('Readability',0),('Feature',8),('Pattern',14),('Stability',16)]
cb=fig.colorbar(im,ax=ax,shrink=0.4,pad=0.01);cb.set_label('dark = top ranks (number: rank)',fontsize=8)
ax.set_title('Person ranking map: rank on each measure (1 = highest value; empty = no recording in that block)',fontsize=11)
plt.savefig('ranking_map.png',dpi=110,bbox_inches='tight');print(len(order),'measure')
