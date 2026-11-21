import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '19_congenital_low_vision_representation'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
r=json.load(open('clv_res.json'))['A_training_source'];INK='#1f2328';MUT='#6e7781'
pi=json.load(open('perm_B2_img.json'));pd=json.load(open('perm_B2_touch.json'))
def pv(s):nl=np.array(s['null']);return (1+np.sum(nl>=s['obs']))/(1+len(nl))
src=[('Congenitally low vision','#e07b39'),('Later vision loss','#2a9d8f'),('Sighted','#2f6fde')]
fig,axs=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
for ax,(cn,st,ttl) in zip(axs,[('B2 touch',pd,'Touch onset (perception)'),('B2 imagery',pi,'Imagining_touched imagery')]):
    c=r[cn]['main'];x=0;ticks=[];labs=[]
    for test in ['Congenitally low vision','Sighted']:
        for s,col in src:
            key=f'{test} ← {s}'
            if key not in c: continue
            d=c[key];ax.bar(x,d['accuracy']*100,0.75,color=col,label=s if test=='Congenitally low vision' else None)
            v=np.array(list(d['person'].values()))*100;ax.scatter(x+np.linspace(-0.2,0.2,len(v)),v,s=10,color=INK,alpha=0.45,zorder=3)
            ax.text(x,d['accuracy']*100+1.2,f"{d['accuracy']*100:.0f}{'*' if d['p']<.05 else ''}",ha='center',fontsize=8.5,color=INK);x+=1
        ticks.append(x-2);labs.append(f'test: {test}\n(n={len(c[test+' ← Sighted']['person'])})');x+=0.8
    ax.axhline(20,color=MUT,ls='--',lw=1);ax.set_xticks(ticks,labs);ax.set_ylim(0,65)
    ax.set_title(f"{ttl}\ntraining from same group − from sighted, difference: {st['obs']*100:+.1f} points, group permutation p={pv(st):.3f}",fontsize=9.5,color=INK)
    for s_ in ['top','right']: ax.spines[s_].set_visible(False)
axs[0].set_ylabel('accuracy (%) · dots: people');axs[0].legend(title='group training the model',frameon=False,fontsize=8,title_fontsize=8,loc='upper left')
fig.suptitle("Tactile imagery of congenitally low vision: which group's brain pattern recognises it? (* = significantly above chance, chance 20%)",fontsize=11,color=INK,y=1.07)
plt.savefig('congenital_low_vision.png',dpi=110,bbox_inches='tight')
