"""Figure 7: double dissociation (A) and occipital alpha during touch (B); Supplementary Figure S5: region x band map."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json, pickle, numpy as np, pandas as pd, mne
from style import *
mne.set_log_level('ERROR')
EPO = os.path.join(ROOT, 'data', 'clean_epochs', 'epo2', 'P15_block2_imagery_motionless-epo.fif')  # channel positions only
C1 = GCOL['Congenitally low vision']; C2 = GCOL['Sighted']
def L(c, s):
    d = json.load(open(res('21_visual_experience_and_regions' if c.startswith('B1') else '20_time_frequency', f'res_{c}_{s}.json')))
    nl = np.array(d['null']); d['p_upper'] = (1 + np.sum(nl >= d['diff'])) / (1 + len(nl)); d['p_lower'] = (1 + np.sum(nl <= d['diff'])) / (1 + len(nl)); return d
fig = plt.figure(figsize=(12.5, 4.9)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.95], wspace=0.3)
ax = fig.add_subplot(gs[0]); bw = 0.36
for i, c in enumerate(['B1_img', 'B2_img']):
    d = L(c, 'alpha')
    ax.bar(i - bw / 2, d['same_group'] * 100, bw * 0.92, color=C1, label='model trained on congenitally low vision' if i == 0 else None)
    ax.bar(i + bw / 2, d['sighted'] * 100, bw * 0.92, color=C2, label='model trained on sighted' if i == 0 else None)
    p = min(d['p_upper'], d['p_lower']); ax.text(i, 38, f"difference {d['diff'] * 100:+.0f} points".replace('-', '−') + f"\np = {p:.3f}", ha='center', fontsize=8.5, color=INK)
ax.axhline(20, color=MUT, ls='--', lw=1)
ax.set_xticks([0, 1], ['Block 1: imagery after\nhearing the word', 'Block 2: imagery after\ntouching the object']); ax.set_ylim(0, 44)
ax.set_ylabel('accuracy in congenitally low vision (%), alpha')
ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, 0.83)); clean(ax)
ax.set_title('A. Same people, two kinds of imagery', fontsize=11, loc='left')
TF = pickle.load(open(res('20_time_frequency', 'tf.pkl'), 'rb'))
k = pd.read_excel(os.path.join(ROOT, 'participants', 'participant_information.xlsx')); GR = dict(zip(k['code'], k['analysis_group']))
info = mne.read_epochs(EPO, preload=False).info; names = info.ch_names
def rel(n): x = np.log(np.nanmean(TF[n]['B2_touch']['P'].reshape(-1, 64, 4), 0))[:, 1]; return x - x.mean()
v = np.mean([rel(n) for n in TF if GR.get(n) == 'Congenitally low vision'], 0) - np.mean([rel(n) for n in TF if GR.get(n) == 'Sighted'], 0)
a = json.load(open(res('21_visual_experience_and_regions', 'activity_res.json')))['touch (relative topography) | Congenitally low vision vs Sighted | alpha'][0]
mask = np.array([c in a['channels'] for c in names])
bx = fig.add_subplot(gs[1]); m = np.abs(v).max()
im, _ = mne.viz.plot_topomap(v, info, axes=bx, show=False, cmap='RdBu_r', vlim=(-m, m), mask=mask,
                             mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', markersize=6))
bx.set_title(f"B. Relative alpha power during touch\ncongenitally low vision minus sighted (cluster p = {a['p']:.3f})", fontsize=11, loc='left')
plt.colorbar(im, ax=bx, shrink=0.75, label='log power difference (blue = lower alpha)')
save(fig, 'Figure7_imagery_and_occipital_alpha')
# ---- S5
r = json.load(open(res('21_visual_experience_and_regions', 'region_group_res.json')))
REG = ['anterior left', 'anterior right', 'mid hat', 'central left', 'central right', 'temporal left', 'temporal right', 'parietal left', 'parietal right', 'occipital (vision)']
LAB = ['frontal left', 'frontal right', 'midline', 'central left', 'central right', 'temporal left', 'temporal right', 'parietal left', 'parietal right', 'occipital']
B = ['theta', 'alpha', 'beta', 'gamma']
M = np.array([[r[f'{g}|{b}']['diff'] * 100 for b in B] for g in REG]); vm = np.abs(M).max()
fig, ax = plt.subplots(figsize=(9.5, 3.8)); im = ax.imshow(M.T, cmap='PuOr_r', vmin=-vm, vmax=vm, aspect='auto')
for i, g in enumerate(REG):
    for j, b in enumerate(B):
        p = r[f'{g}|{b}']['p_cell']; ax.text(i, j, f"{M[i, j]:+.0f}" + ('*' if p < .05 else ''), ha='center', va='center', fontsize=8.5, color=INK)
ax.set_xticks(range(len(REG)), LAB, rotation=35, ha='right'); ax.set_yticks(range(4), B)
for s in ax.spines.values(): s.set_visible(False)
plt.colorbar(im, ax=ax, shrink=0.85, label='own-group minus sighted training\n(accuracy points)')
ax.set_title('Congenitally low vision, Block 2 imagery after touch: regions and frequency bands\n(* uncorrected cell p < 0.05; none survives correction for 40 cells)', fontsize=10, loc='left')
save(fig, 'FigureS5_region_by_band')
