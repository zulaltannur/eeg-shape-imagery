import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '21_visual_experience_and_regions'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json,numpy as np,mne,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;mne.set_log_level('ERROR')
info=mne.read_epochs('../../data/clean_epochs/epo2/P15_block2_imagery_motionless-epo.fif',preload=False).info;names=info.ch_names
W=[('B1_erp','Block 1 · word onset'),('B1_img','Block 1 · imagining what was heard'),('B2_touch','Block 2 · touch'),('B2_img','Block 2 · imagining what was touched'),('B3_erp','Block 3 · picture onset'),('B3_perc','Block 3 · picture viewing'),('B3_img','Block 3 · imagining what was seen')]
fig,axs=plt.subplots(1,7,figsize=(21,3.6))
for ax,(c,t) in zip(axs,W):
    r=json.load(open(f'../../results/21_visual_experience_and_regions/sl_{c}.json'));H=r['cell']
    v=np.array([max(H[k]['accuracy'] for k in H if k.split('|')[0]==ch) for ch in names])*100
    best=r['clusters'][0] if r['clusters'] else None;mask=np.array([best is not None and ch in best['channels'] for ch in names])
    im,_=mne.viz.plot_topomap(v,info,axes=ax,show=False,cmap='Oranges',vlim=(18,30),mask=mask,mask_params=dict(marker='o',markerfacecolor='w',markeredgecolor='k',markersize=5))
    ax.set_title(f"{t}\nstrongest cluster: {best['band_name']} · p={best['p_cluster']:.3f}" if best else t,fontsize=8.5)
fig.colorbar(im,ax=axs,shrink=0.7,label='local shape accuracy (%), highest of 4 bands · chance 20')
fig.suptitle('Where on the scalp is shape information? A separate model for each channel and its neighbours (leave-one-subject-out); white dots are the strongest cluster',fontsize=11)
plt.savefig('scan_map.png',dpi=105,bbox_inches='tight')
