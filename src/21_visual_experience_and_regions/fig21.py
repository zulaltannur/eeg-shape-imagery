import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '21_visual_experience_and_regions'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json,numpy as np,pickle,mne,pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;mne.set_log_level('ERROR')
INK='#1f2328';MUT='#6e7781';C1='#e07b39';C2='#2f6fde'
def L(c,s):
    d=json.load(open(f'../../results/{"21_visual_experience_and_regions" if c.startswith("B1") else "20_time_frequency"}/res_{c}_{s}.json'));nl=np.array(d['null']);d['p_upper']=(1+np.sum(nl>=d['diff']))/(1+len(nl));d['p_lower']=(1+np.sum(nl<=d['diff']))/(1+len(nl));return d
fig=plt.figure(figsize=(17,4.8));gs=fig.add_gridspec(1,3,width_ratios=[1,0.9,1.25],wspace=0.35)
ax=fig.add_subplot(gs[0]);bw=0.36
for i,(c,ttl) in enumerate([('B1_img','Block 1: imagining what\nwas heard'),('B2_img','Block 2: imagining what\nwas touched')]):
    d=L(c,'alpha')
    ax.bar(i-bw/2,d['same_group']*100,bw*0.92,color=C1,label='trained on congenitally low vision' if i==0 else None)
    ax.bar(i+bw/2,d['sighted']*100,bw*0.92,color=C2,label='trained on sighted' if i==0 else None)
    p=min(d['p_upper'],d['p_lower']);ax.text(i,39,f"diff {d['diff']*100:+.0f} points\np={p:.3f}",ha='center',fontsize=8.5,color=INK)
ax.axhline(20,color=MUT,ls='--',lw=1);ax.set_xticks([0,1],['Block 1: imagining what\nwas heard','Block 2: imagining what\nwas touched']);ax.set_ylim(0,42);ax.set_ylabel('accuracy in congenitally low vision (%) · alpha')
ax.legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(0,0.84));ax.set_title('Same people, two kinds of imagery: double dissociation',fontsize=10,color=INK)
for s in ['top','right']: ax.spines[s].set_visible(False)
# occipital alpha cluster (touch)
TF=pickle.load(open('../../results/20_time_frequency/tf.pkl','rb'));k=pd.read_excel('../../participants/participant_information.xlsx');GR=dict(zip(k['code'],k['analysis_group']))
info=mne.read_epochs('../../data/clean_epochs/epo2/P15_block2_imagery_motionless-epo.fif',preload=False).info;names=info.ch_names
def rel(n):x=np.log(np.nanmean(TF[n]['B2_touch']['P'].reshape(-1,64,4),0))[:,1];return x-x.mean()
v=np.mean([rel(n) for n in TF if GR.get(n)=='Congenitally low vision'],0)-np.mean([rel(n) for n in TF if GR.get(n)=='Sighted'],0)
a=json.load(open('../../results/21_visual_experience_and_regions/activity_res.json'))['touch (relative topography) | Congenitally low vision vs Sighted | alpha'][0]
mask=np.array([c in a['channels'] for c in names])
ax=fig.add_subplot(gs[1]);m=np.abs(v).max();im,_=mne.viz.plot_topomap(v,info,axes=ax,show=False,cmap='RdBu_r',vlim=(-m,m),mask=mask,mask_params=dict(marker='o',markerfacecolor='w',markeredgecolor='k',markersize=6))
ax.set_title(f"Relative alpha power during touch\ncongenitally low vision − sighted (cluster p={a['p']:.3f})",fontsize=10,color=INK);plt.colorbar(im,ax=ax,shrink=0.7,label='log power difference (blue: lower alpha)')
# region × band
r=json.load(open('../../results/21_visual_experience_and_regions/region_group_res.json'));REG=['anterior left','anterior right','mid hat','central left','central right','temporal left','temporal right','parietal left','parietal right','occipital (vision)'];B=['theta','alpha','beta','gamma']
M=np.array([[r[f'{g}|{b}']['diff']*100 for b in B] for g in REG]);ax=fig.add_subplot(gs[2]);vm=np.abs(M).max()
im=ax.imshow(M.T,cmap='PuOr_r',vmin=-vm,vmax=vm,aspect='auto')
for i,g in enumerate(REG):
    for j,b in enumerate(B):
        p=r[f'{g}|{b}']['p_cell'];ax.text(i,j,f"{M[i,j]:+.0f}"+('*' if p<.05 else ''),ha='center',va='center',fontsize=8,color=INK)
ax.set_xticks(range(len(REG)),REG,rotation=40,ha='right');ax.set_yticks(range(4),B);plt.colorbar(im,ax=ax,shrink=0.8,label='same group − from sighted (points)')
ax.set_title('The distinctive pattern in tactile imagery by region\n(* cell p<0.05; after correction for 40 cells none is significant on its own)',fontsize=9.5,color=INK)
fig.suptitle('Congenitally low vision: the form of imagery changes with the sense, the visual region takes part in touch',fontsize=12,color=INK,y=1.04)
plt.savefig('stage21.png',dpi=110,bbox_inches='tight')
