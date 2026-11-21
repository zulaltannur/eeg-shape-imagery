import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '15_shape_generation_every_block'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,json,sys,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
sys.path.insert(0,'../../src/09_superquadric_bridge');from sq import PAR,surface
from matplotlib.colors import LightSource
CL=['cone','cube','sphere','pyramid','cylinder'];r=json.load(open('gen_res.json'));ev=json.load(open('gen_eval.json'))
order=['Block 1 · word onset','Block 1 · imagery','Block 2 · touch','Block 2 · touch to imagery','Block 3 · picture onset','Block 3 · imagery']
LS=LightSource(azdeg=315,altdeg=40);SURF={c:surface(*PAR[c],n=36) for c in CL}
fig=plt.figure(figsize=(19,13))
outer=fig.add_gridspec(2,3,hspace=0.28,wspace=0.12)
for b,name in enumerate(order):
    res=r[name];Y=np.concatenate([res[n]['y'] for n in res]);Pd=np.concatenate([np.array(CL)[np.array(res[n]['P']).argmax(1)] for n in res])
    M=np.array([[np.mean(Pd[Y==t]==g) for g in CL] for t in CL]);e=ev[name]
    inner=outer[b//3,b%3].subgridspec(6,6,hspace=0.02,wspace=0.02)
    for i,t in enumerate(CL):
        ax=fig.add_subplot(inner[i+1,0]);ax.axis('off');ax.text(0.95,0.5,t,ha='right',va='center',fontsize=9)
        for j,gg in enumerate(CL):
            if i==0: a2=fig.add_subplot(inner[0,j+1]);a2.axis('off');a2.text(0.5,0.1,gg,ha='center',va='bottom',fontsize=9)
            ax=fig.add_subplot(inner[i+1,j+1],projection='3d');X,Yy,Z=SURF[gg]
            al=float(np.clip(M[i,j]/0.45,0.06,1));col=(0.72,0.12,0.10) if i==j else (0.25,0.25,0.25)
            ax.plot_surface(X,Yy,Z,color=col,alpha=al,shade=True,lightsource=LS,linewidth=0,antialiased=False)
            ax.set_box_aspect((1,1,1));L=0.6;ax.set_xlim(-L,L);ax.set_ylim(-L,L);ax.set_zlim(-L,L);ax.view_init(18,35);ax.set_axis_off()
            ax.text2D(0.5,-0.02,f'%{M[i,j]*100:.0f}',transform=ax.transAxes,ha='center',fontsize=7,color='#b71c1c' if i==j else '#555')
    ax=fig.add_subplot(outer[b//3,b%3]);ax.axis('off')
    star='★ ' if e['p_acc']<.05 else ''
    ax.set_title(f"{star}{name}\ncorrect generation %{e['acc']*100:.1f} (chance 20%, p={e['p_acc']:.3f})\ncoordinate distance p={e['p_chamfer']:.3f}",fontsize=10)
fig.text(0.02,0.5,'TRUE SHAPE (rows)',rotation=90,va='center',fontsize=11);fig.text(0.5,0.955,'SHAPE GENERATED FROM EEG (columns)',ha='center',fontsize=11)
fig.suptitle("Shapes generated from EEG for every block — in never-seen people, the shape the model generated on each trial\nRed = correct shape generated; the shape's opacity shows how often it was generated",fontsize=12,y=1.0)
plt.savefig('generation_matrix.png',dpi=90,bbox_inches='tight')
