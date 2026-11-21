import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '23_three_groups_cases_connectivity'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json,numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
r=json.load(open('connectivity_res.json'))['occipital_second'];CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
k=pd.read_excel('../../participants/participant_information.xlsx');G0=dict(zip(k['code'],k['analysis_group']))
G=lambda n:'CLV' if n in CLV else ('LVL' if n in LVL else ('Sighted' if G0.get(n)=='Sighted' else None))
INK='#1f2328';col={'Sighted':'#2f6fde','LVL':'#2a9d8f','CLV':'#e07b39'};lab={'Sighted':'Sighted','LVL':'Typical vision then lost','CLV':'Congenitally limited vision'}
fig,axs=plt.subplots(1,2,figsize=(12,4.2),sharey=True)
for ax,(c,t) in zip(axs,[('B2_touch','While_touching'),('B2_img','Imagining_touched while_imagining')]):
    x=np.arange(7)+0.5
    for g in ['Sighted','LVL','CLV']:
        M=np.array([r[c][n][:7] for n in r[c] if G(n)==g]);m=M.mean(0);se=M.std(0,ddof=1)/np.sqrt(len(M))
        ax.plot(x,m,'-o',color=col[g],lw=2,ms=5,label=f'{lab[g]} (n={len(M)})');ax.fill_between(x,m-se,m+se,color=col[g],alpha=0.15,lw=0)
    for case,ls in [('P10',':'),('P23','--')]:
        ax.plot(x,r[c][case][:7],ls,color='#57606a',lw=1.3,label=case)
    ax.set_title(t,fontsize=10.5,color=INK);ax.set_xlabel('second');ax.set_xticks(range(8))
    for s in ['top','right']: ax.spines[s].set_visible(False)
axs[0].set_ylabel('occipital alpha − whole-head mean (log)\nlower = occipital more active');axs[0].legend(frameon=False,fontsize=8,loc='upper right',bbox_to_anchor=(1,0.75))
fig.suptitle('Visual cortex alpha second by second (blindfolded): order congenitally limited < typical vision then lost < sighted\n(ordered trend: touch p=0.003, imagery p=0.011)',fontsize=11,color=INK,y=1.08)
plt.savefig('occipital_course.png',dpi=110,bbox_inches='tight')
