import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '20_time_frequency'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json,numpy as np,pickle,pandas as pd,mne,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;mne.set_log_level('ERROR')
INK='#1f2328';MUT='#6e7781';C1='#e07b39';C2='#2f6fde'
def L(c,s):
    d=json.load(open(f'res_{c}_{s}.json'));nl=np.array(d['null']);d['p']=(1+np.sum(nl>=d['diff']))/(1+len(nl));return d
fig=plt.figure(figsize=(15,4.6));gs=fig.add_gridspec(1,3,width_ratios=[1.3,1,0.9],wspace=0.3)
ax=fig.add_subplot(gs[0]);W=['z0-2','z1-3','z2-4','z3-5','z4-6','z5-7'];x=np.arange(6)
for key,col,lab in [('same_group',C1,'trained on congenitally low vision'),('sighted',C2,'trained on sighted')]:
    ax.plot(x,[L('B2_img',w)[key]*100 for w in W],'-o',color=col,lw=2,ms=6,label=lab)
for i,w in enumerate(W): d=L('B2_img',w);ax.text(i,36,f"p={d['p']:.2f}",ha='center',fontsize=8,color=INK if d['p']<.1 else MUT)
ax.axhline(20,color=MUT,ls='--',lw=1);ax.set_xticks(x,[w[1:].replace('-','–')+' s' for w in W]);ax.set_ylim(5,40);ax.set_ylabel('accuracy (%)')
ax.set_title('Time: which seconds of imagery? (2 s window, 1 s step)',fontsize=10,color=INK);ax.legend(frameon=False,fontsize=8,loc='lower left')
ax=fig.add_subplot(gs[1]);B=['theta','alpha','beta','gamma'];x=np.arange(4);bw=0.38
for k,(key,col) in enumerate([('same_group',C1),('sighted',C2)]):
    ax.bar(x+(k-0.5)*bw,[L('B2_img',b)[key]*100 for b in B],bw*0.92,color=col)
for i,b in enumerate(B): d=L('B2_img',b);ax.text(i,36,f"p={d['p']:.3f}",ha='center',fontsize=8,color=INK if d['p']<.05 else MUT,weight='bold' if d['p']<.05 else None)
ax.axhline(20,color=MUT,ls='--',lw=1);ax.set_xticks(x,['theta\n4–7 Hz','alpha\n8–13 Hz','beta\n14–25 Hz','gamma\n26–40 Hz']);ax.set_ylim(0,40)
ax.set_title('Frequency: which brain rhythm? (whole imagery window)',fontsize=10,color=INK)
for a in fig.axes:
    for s in ['top','right']: a.spines[s].set_visible(False)
# alpha power topography: congenitally low vision − sighted (imagery)
TF=pickle.load(open('tf.pkl','rb'));k=pd.read_excel('../../participants/participant_information.xlsx');G=dict(zip(k['code'],k['analysis_group']))
def amap(n,c):P=TF[n][c]['P'][:,:,:,1];return np.log(np.nanmean(P.reshape(-1,64),0))
dif=lambda c:np.mean([amap(n,c) for n in TF if G.get(n)=='Congenitally low vision'],0)-np.mean([amap(n,c) for n in TF if G.get(n)=='Sighted'],0)
ep=mne.read_epochs('../../data/clean_epochs/epo2/P15_block2_imagery_motionless-epo.fif',preload=False);info=ep.info
v=dif('B2_img')-dif('B2_touch')
ax=fig.add_subplot(gs[2]);im,_=mne.viz.plot_topomap(v,info,axes=ax,show=False,cmap='RdBu_r',vlim=(-np.abs(v).max(),np.abs(v).max()))
ax.set_title('Alpha power: increase in imagery relative to touch\ndifference (congenitally low vision − sighted)',fontsize=10,color=INK);plt.colorbar(im,ax=ax,shrink=0.7,label='log power difference')
fig.suptitle('Tactile imagery pattern specific to congenitally low vision: time and frequency',fontsize=12,color=INK,y=1.03)
plt.savefig('time_frequency.png',dpi=110,bbox_inches='tight')
print(np.round(v,2)[:5],[info.ch_names[i] for i in np.argsort(v)[-6:]],[info.ch_names[i] for i in np.argsort(v)[:6]])
