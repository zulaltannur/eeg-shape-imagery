"""Supplementary S1: THINGS-EEG2 pretrained encoder; S2: searchlight scan maps."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json, numpy as np, mne
from style import *
mne.set_log_level('ERROR')
R = json.load(open(res('07_things_eeg2_pretraining', 'apply_res.json')))
NAME = {'B1_word': 'Block 1, word onset', 'B1_img': 'Block 1, imagery', 'B2_touch': 'Block 2, touch', 'B2_img': 'Block 2, imagery',
        'B3_picture': 'Block 3, picture onset', 'B3_viewing': 'Block 3, picture viewing', 'B3_img': 'Block 3, imagery'}
keys = [k for k in R if k.startswith('finetuned|')]
def label(k):
    a, b = k.split('|')[1].split('->')
    return NAME[a] if a == b else f"{NAME[a].split(', ')[1]} → {NAME[b]}"
fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(keys) + 1.4)); y = np.arange(len(keys))[::-1]
for yy, k in zip(y, keys):
    r = R[k]; acc = r['acc'] * 100; sig = r['p'] < 0.05
    ax.barh(yy, acc, color=GCOL['Sighted'] if sig else '#b8c0c8', height=0.62)
    ax.plot([r['null95'] * 100] * 2, [yy - 0.33, yy + 0.33], color=INK, lw=1.8)
    ax.text(max(acc, r['null95'] * 100) + 0.6, yy, f"{acc:.1f}%  (p = {r['p']:.3f})", va='center', fontsize=8.5, color=INK, weight='bold' if sig else None)
ax.set_yticks(y, [label(k) for k in keys], fontsize=9); ax.axvline(20, color=MUT, ls='--', lw=1); ax.set_xlim(0, 40); clean(ax)
ax.set_xlabel('accuracy in never-seen people (%); dashed line: chance 20%; black tick: 95th percentile of the permutation null')
ax.set_title('Encoder pretrained on THINGS-EEG2, fine-tuned with leave-one-subject-out validation', fontsize=10.5, loc='left')
save(fig, 'FigureS1_things_encoder')
info = mne.read_epochs(os.path.join(ROOT, 'data', 'clean_epochs', 'epo2', 'P15_block2_imagery_motionless-epo.fif'), preload=False).info; names = info.ch_names
W = [('B1_erp', 'Block 1, word onset'), ('B1_img', 'Block 1, imagery'), ('B2_touch', 'Block 2, touch'), ('B2_img', 'Block 2, imagery'),
     ('B3_erp', 'Block 3, picture onset'), ('B3_perc', 'Block 3, picture viewing'), ('B3_img', 'Block 3, imagery')]
fig, axs = plt.subplots(2, 4, figsize=(15, 8)); axs = axs.ravel(); axs[-1].axis('off')
for ax, (c, t) in zip(axs, W):
    r = json.load(open(res('21_visual_experience_and_regions', f'sl_{c}.json'))); H = r['cell']
    v = np.array([max(H[k]['accuracy'] for k in H if k.split('|')[0] == ch) for ch in names]) * 100
    best = r['clusters'][0] if r['clusters'] else None; mask = np.array([best is not None and ch in best['channels'] for ch in names])
    im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap='Oranges', vlim=(18, 30), mask=mask,
                                 mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', markersize=5))
    ax.set_title(f"{t}\nstrongest cluster: {best['band_name']}, p = {best['p_cluster']:.3f}" if best else t, fontsize=9.5)
fig.colorbar(im, ax=axs[-1], location='left', shrink=0.7, label='local shape accuracy (%), best of four bands; chance 20%')
save(fig, 'FigureS2_searchlight')
