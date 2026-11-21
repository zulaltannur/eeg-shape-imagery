import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '18_detailed_analyses'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
r=json.load(open('detail_res.json'));CL=['cone','cube','sphere','pyramid','cylinder']
INK='#1f2328';MUT='#6e7781'
plt.rcParams.update({'font.size':9,'axes.edgecolor':'#d0d7de','axes.labelcolor':INK,'xtick.color':MUT,'ytick.color':MUT})
# 1) confusion matrices
ws=['Block 1 · imagery','Block 2 · touch to imagery','Block 3 · picture onset','Block 3 · imagery']
fig,ax=plt.subplots(1,4,figsize=(17,4.6))
for a,w in zip(ax,ws):
    M=np.array(r['A_confusion_geometry'][w]['confusion']);im=a.imshow(M,cmap='Blues',vmin=0.14,vmax=0.30)
    for i in range(5):
        for j in range(5): a.text(j,i,f'{M[i,j]*100:.0f}',ha='center',va='center',fontsize=8,color='white' if M[i,j]>0.26 else INK)
    a.set_xticks(range(5),CL,rotation=40);a.set_yticks(range(5),CL);a.set_xlabel('model probability (%)');a.set_ylabel('true shape') if w==ws[0] else None
    A=r['A_confusion_geometry'][w];a.set_title(f"{w}\ncurved/cornered %{A['curved_accuracy']*100:.0f} (p={A['curved_p']:.3f})\npointed %{A['pointed_accuracy']*100:.0f} (p={A['pointed_p']:.3f})",fontsize=8.5,color=INK)
plt.subplots_adjust(wspace=0.45);fig.colorbar(im,ax=ax,shrink=0.8,label='mean probability')
fig.suptitle('Which shapes are confused with each other? (all people, calibrated probabilities; chance = 20%, attribute chance = 50%)',fontsize=11,color=INK)
plt.savefig('confusion_matrices.png',dpi=110,bbox_inches='tight');plt.close()
# 2) groups
ws=['Block 1 · imagery','Block 2 · imagery','Block 2 · touch to imagery','Block 3 · picture onset','Block 3 · imagery']
G=['Sighted','Congenitally low vision','Later vision loss','Congenitally blind'];col=['#2f6fde','#e07b39','#2a9d8f','#8a63d2']
fig,ax=plt.subplots(figsize=(12,4.2));x=np.arange(len(ws));bw=0.2
for k,(g,c) in enumerate(zip(G,col)):
    for i,w in enumerate(ws):
        d=r['C_groups'][w]['groups'].get(g)
        if not d: continue
        xx=x[i]+(k-1.5)*bw;ax.bar(xx,d['accuracy']*100,bw*0.9,color=c,label=g if i==0 else None)
        ax.text(xx,d['accuracy']*100+0.6,f"{d['accuracy']*100:.0f}{'*' if d['p_chance']<.05 else ''}\nn={d['n']}",ha='center',fontsize=7,color=INK)
ax.axhline(20,color=MUT,ls='--',lw=1);ax.text(len(ws)-0.5,20.5,'chance 20%',color=MUT,fontsize=8,ha='right')
ax.set_xticks(x,[w.replace(' · ','\n')+f"\ngroup difference p={r['C_groups'][w]['group_diff_p']:.3f}" for w in ws]);ax.set_ylabel('accuracy (%)');ax.set_ylim(0,40)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.legend(frameon=False,ncol=4,loc='upper left');ax.set_title('Accuracy by vision group (* = significantly above chance within the group)',fontsize=11,color=INK)
plt.savefig('groups.png',dpi=110,bbox_inches='tight');plt.close()
