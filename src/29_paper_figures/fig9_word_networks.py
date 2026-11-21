"""Figure 9: hearing the shape's name (Block 1, exact word onsets). (A) theta power change by region and group;
(B) co-variation of auditory theta with each region; (C) mean co-variation per person, ordered by visual experience."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json, numpy as np
from style import *
O = json.load(open(res('27_sound_detailed', 'region_network_res_v2.json')))
CLV = {'P01', 'P15', 'P02', 'P13', 'P17', 'P06', 'P03', 'P10', 'P23'}; LVL = {'P18', 'P21', 'P16', 'P20'}
G = lambda n: 'CLV' if n in CLV else ('LVL' if n in LVL else 'Sighted')
GN = [('CLV', 'Congenitally limited vision', GCOL['Congenitally low vision']), ('LVL', 'Later vision loss', GCOL['Later vision loss']), ('Sighted', 'Sighted', GCOL['Sighted'])]
REG = ['auditory (temporal)', 'language (left fronto-temporal)', 'fronto-central', 'hand-touch (central)', 'parietal', 'vision (occipital)']
LAB = ['auditory', 'language\n(left)', 'fronto-\ncentral', 'somato-\nsensory', 'parietal', 'visual']
fig = plt.figure(figsize=(16, 4.9)); gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.1, 0.7], wspace=0.28)
ax = fig.add_subplot(gs[0]); x = np.arange(len(REG)); bw = 0.26
for k, (g, gl, c) in enumerate(GN):
    ax.bar(x + (k - 1) * bw, [O[f'change | {r} | theta'][g] for r in REG], bw * 0.92, color=c, label=gl)
for i, r in enumerate(REG):
    p = O[f'change | {r} | theta']['p_everyone']; ax.text(i, 0.12, f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001', ha='center', fontsize=7.5, color=INK)
ax.axhline(0, color=GRID, lw=1); ax.set_xticks(x, LAB, fontsize=8); ax.set_ylabel('theta power change after the word (dB)')
ax.set_ylim(-1.4, 0.3); ax.legend(frameon=False, fontsize=8, loc='lower left'); clean(ax)
ax.set_title('A. Every region responds to the word (theta, all participants)', fontsize=10.5, loc='left')
bx = fig.add_subplot(gs[1]); R5 = REG[1:]
for k, (g, gl, c) in enumerate(GN):
    bx.bar(np.arange(5) + (k - 1) * bw, [O[f'coupling | auditory theta ↔ {r} theta'][g] for r in R5], bw * 0.92, color=c)
bx.set_xticks(np.arange(5), LAB[1:], fontsize=8); bx.set_ylabel('trial-level co-variation with auditory theta (ρ)'); clean(bx)
bx.set_title('B. Co-variation of the auditory region\nwith each other region (theta)', fontsize=10.5, loc='left')
S = O['summary | mean co-variation of auditory theta with the other 5 regions']; k_ = S['person']
cx = fig.add_subplot(gs[2])
for i, (g, gl, c) in enumerate(GN):
    v = [k_[n] for n in k_ if G(n) == g]
    cx.bar(i, np.mean(v), 0.6, color=c); cx.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=13, color=INK, alpha=0.5, zorder=3)
    cx.text(i, np.mean(v) + 0.01, f'{np.mean(v):.2f}', ha='center', va='bottom', fontsize=8.5, color=INK, bbox=dict(fc='white', ec='none', pad=0.5))
cx.set_xticks(range(3), ['Congenitally\nlimited vision', 'Later\nvision loss', 'Sighted'], fontsize=8); cx.set_ylabel('mean co-variation (ρ), dots = people'); clean(cx)
cx.set_title(f"C. Ordered by visual experience\n(ordered trend p = {S['p_ordered']:.3f})", fontsize=10.5, loc='left')
save(fig, 'Figure9_word_networks')
