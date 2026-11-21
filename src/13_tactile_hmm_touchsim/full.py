import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '13_tactile_hmm_touchsim'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,pickle,json,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from hmm import *
M=fit(NAMES,iters=30);pickle.dump(M,open('hmm_full.pkl','wb'))
keys=D[NAMES[0]]['keys']
occ={};dwell={};paths={}
for n in NAMES:
    O=[];Oo=[];Oe=[]
    for t in D[n]['trials']:
        s=t['shape'];m=MASK[s];L=loglik_em(t['F'],t['ok'],M['mu'],M['present'])
        _,g,_=fb(L,masked_log(M['A'][s],m),np.log(M['pi'][s]*m+1e-12)-np.log((M['pi'][s]*m).sum()),m)
        st=g.argmax(1);paths[(n,t['trial'])]=(s,st)
        oc=np.bincount(st,minlength=K)/len(st);O.append(oc);(Oo if t['trial']%2 else Oe).append(oc)
    occ[n]=(np.mean(O,0),np.mean(Oo,0),np.mean(Oe,0))
# reliability of the person-specific pattern: do state occupancy ratios in odd/even trials rank people in the same order
for k in range(K):
    a=[occ[n][1][k] for n in NAMES];b=[occ[n][2][k] for n in NAMES];r=spearmanr(a,b).correlation
    print(f'{ST[k]:15s} between-person occupancy odd/even reliability r={r:+.2f}')
np.set_printoptions(precision=2,suppress=True)
print('EEG signatures of the states (z):');print(keys);print(M['mu'])
fig,ax=plt.subplots(1,2,figsize=(15,4.5),gridspec_kw={'width_ratios':[1.2,1]})
im=ax[0].imshow(M['mu'],cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto');ax[0].set_yticks(range(K));ax[0].set_yticklabels(ST)
ax[0].set_xticks(range(len(keys)));ax[0].set_xticklabels([k.replace('_',' ') for k in keys],rotation=60,ha='right',fontsize=8);plt.colorbar(im,ax=ax[0],label='relative to the mean (z)')
ax[0].set_title('Shared EEG signature of each contact state (learned from all people)')
O=np.array([occ[n][0] for n in NAMES]);im2=ax[1].imshow(O.T,cmap='Reds',aspect='auto');ax[1].set_yticks(range(K));ax[1].set_yticklabels(ST)
ax[1].set_xticks(range(len(NAMES)));ax[1].set_xticklabels(NAMES,rotation=70,fontsize=8);plt.colorbar(im2,ax=ax[1],label='share of time')
ax[1].set_title('Person-specific exploration pattern: share of time spent in each state');plt.tight_layout();plt.savefig('hmm_states.png',dpi=110)
fig,axs=plt.subplots(5,1,figsize=(13,7),sharex=True);cols=['#bdbdbd','#64b5f6','#e53935','#7b1fa2','#ffb300']
for i,s in enumerate(CL):
    ex=[(k,v) for k,v in paths.items() if v[0]==s][:6]
    for j,((n,dn),(_,st)) in enumerate(ex):
        for tt,x in enumerate(st): axs[i].add_patch(plt.Rectangle((tt*0.25,j),0.25,0.9,color=cols[x]))
    axs[i].set_xlim(0,12);axs[i].set_ylim(0,6);axs[i].set_yticks(np.arange(6)+.45);axs[i].set_yticklabels([n for (n,_),_ in ex],fontsize=7);axs[i].set_ylabel(s,rotation=0,ha='right',fontsize=10)
axs[-1].set_xlabel('Time since taking the shape (s)')
from matplotlib.patches import Patch;axs[0].legend(handles=[Patch(color=c,label=l) for c,l in zip(cols,ST)],ncol=5,loc='lower left',bbox_to_anchor=(0,1.02),fontsize=8)
plt.suptitle('Exploration paths the model derived from EEG (each row a trial)',y=1.04);plt.tight_layout();plt.savefig('hmm_paths.png',dpi=110,bbox_inches='tight')
json.dump({n:occ[n][0].tolist() for n in NAMES},open('hmm_occ.json','w'),ensure_ascii=False)
