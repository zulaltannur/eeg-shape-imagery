import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '23_three_groups_cases_connectivity'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import pickle,numpy as np,sys,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
sys.path.insert(0,'../../src/09_superquadric_bridge');from sq import PAR,draw
P,yy=pickle.load(open('../../results/23_three_groups_cases_connectivity/groupP.pkl','rb'));CL=['cone','cube','sphere','pyramid','cylinder'];INK='#1f2328';MUT='#6e7781'
cols=list(P);fig=plt.figure(figsize=(2.6*5,2.0*5+1.2))
for i,c in enumerate(CL):
    draw(fig.add_subplot(5,5,i*5+1,projection='3d'),PAR[c],('TRUE SHAPE\n' if i==0 else '')+c)
    for j,lab in enumerate(cols):
        ax=fig.add_subplot(5,5,i*5+j+2);pr=np.vstack([P[lab][n][yy[n]==i] for n in P[lab]]).mean(0)*100
        hit=np.mean(np.concatenate([P[lab][n][yy[n]==i].argmax(1)==i for n in P[lab]]))*100
        ax.bar(range(5),pr,color=['#e07b39' if k==i else '#c9d1d9' for k in range(5)],width=0.75)
        ax.axhline(20,color=MUT,ls='--',lw=0.8);ax.set_ylim(10,30);ax.set_xticks(range(5),[x[:3] for x in CL],fontsize=7)
        ax.tick_params(axis='y',labelsize=7);[ax.spines[s].set_visible(False) for s in ['top','right']]
        acc=np.mean([np.mean(P[lab][n].argmax(1)==yy[n]) for n in P[lab]])*100
        ax.set_title((f"{lab}\noverall accuracy %{acc:.0f}\n\n" if i==0 else '')+f"hit_rate %{hit:.0f}",fontsize=8,color=INK)
fig.suptitle("Imagining what was touched (alpha): the model's mean probabilities by true shape (orange = correct shape; dashed line = chance 20%)\nEach group's imagery, by which group's brain pattern it is read with · the model never saw the person being read",fontsize=10.5,color=INK)
plt.tight_layout(rect=(0,0,1,0.95));plt.savefig('../../results/23_three_groups_cases_connectivity/group_shapes.png',dpi=100)
