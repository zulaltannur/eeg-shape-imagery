import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '27_sound_detailed'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json,numpy as np,pickle,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
O=json.load(open('region_network_res.json'));t=np.load('region_network_t.npy');TC=pickle.load(open('region_network_tc.pkl','rb'))
CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
G=lambda n:'CLV' if n in CLV else ('LVL' if n in LVL else 'Sighted');col={'Sighted':'#2f6fde','LVL':'#2a9d8f','CLV':'#e07b39'};lab={'Sighted':'Sighted','LVL':'Typical vision then lost','CLV':'Congenitally limited vision'}
fig,axs=plt.subplots(1,3,figsize=(16,4.3),gridspec_kw=dict(width_ratios=[1,1,0.9]))
for ax,key,ttl in [(axs[0],'auditory (temporal)|theta','Auditory region · theta'),(axs[1],'vision (occipital)|alpha','Vision region (occipital) · alpha')]:
    m=(t>=-0.3)&(t<=1.3)
    for g in ['Sighted','LVL','CLV']:
        X=np.array([TC[n][key] for n in TC if G(n)==g]);ax.plot(t[m],X.mean(0)[m],color=col[g],lw=2,label=f'{lab[g]} (n={len(X)})')
    ax.axvline(0,color='#57606a',lw=1,ls='--');ax.axhline(0,color='#d0d7de',lw=1);ax.set_xlabel('after word onset (s)');ax.set_ylabel('power change (dB, relative to the silence before the beep)');ax.set_title(ttl,fontsize=10)
    for s in ['top','right']: ax.spines[s].set_visible(False)
axs[0].legend(frameon=False,fontsize=8)
k=O['summary | mean co-variation of auditory theta with the other 5 regions']['person'];ax=axs[2]
for i,g in enumerate(['CLV','LVL','Sighted']):
    v=[k[n] for n in k if G(n)==g];ax.bar(i,np.mean(v),0.6,color=col[g]);ax.scatter(np.full(len(v),i)+np.linspace(-0.15,0.15,len(v)),v,s=12,color='#1f2328',alpha=0.5,zorder=3)
ax.text(0,k['P10']+0.02,'P10',fontsize=8,ha='center')
ax.set_xticks(range(3),['Congenital\nlimited','Typical vision\nthen lost','Sighted']);ax.set_ylabel('trial-level co-variation (ρ, rho)')
ax.set_title(f"How much does the auditory region work together with the other regions?\nordered trend p={O['summary | mean co-variation of auditory theta with the other 5 regions']['p_ordered']:.3f}",fontsize=10)
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.suptitle("When the shape's name is heard: all regions engage; in congenitally limited vision the regions are more independent of each other",fontsize=11.5,y=1.03)
plt.savefig('word_networks.png',dpi=110,bbox_inches='tight')
