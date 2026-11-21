"""Figure 8: (A) pairwise similarity of tactile imagery vs closeness in years of typical vision (Mantel);
(B, C) occipital alpha second by second during touch and imagery, group means only."""
import os, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); _RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
os.chdir(os.path.join(HERE, '..', '..', 'results', '21_visual_experience_and_regions'))
code = open('../../src/21_visual_experience_and_regions/mantel.py').read()
exec(code.split('out={}')[0])
import sys; sys.path.insert(0, HERE); from style import *
R = feats('B2_img', ['alpha']); ns, S = pair(R); Ss = (S + S.T) / 2; iu = np.triu_indices(len(ns), 1)
v = np.array([YRS[n] for n in ns], float); dist = np.abs(v[:, None] - v[None])[iu]; sim = Ss[iu]
rho = spearmanr(sim, -dist).correlation
ref = json.load(open('mantel_res.json'))['B2 imagery · alpha']['years with typical vision']; assert abs(rho - ref['rho']) < 0.002, rho
fig = plt.figure(figsize=(15, 4.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.28)
ax = fig.add_subplot(gs[0]); rng2 = np.random.default_rng(1)
ax.scatter(dist + rng2.uniform(-0.4, 0.4, len(dist)), sim * 100, s=14, color=MUT, alpha=0.5, lw=0)
edges = [0, 0.5, 10, 20, 30, 45, 70]; cx = []; cy = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (dist >= lo) & (dist < hi)
    if m.sum() >= 5: cx.append(dist[m].mean()); cy.append(sim[m].mean() * 100)
ax.plot(cx, cy, '-o', color=INK, lw=2, ms=7, markeredgecolor='white', markeredgewidth=1.5, label='mean within distance bins')
ax.axhline(20, color=MUT, ls='--', lw=1)
ax.set_xlabel('difference in years lived with typical vision (pair of people)'); ax.set_ylabel('probability given to the correct shape (%)\nmodel trained on one person, tested on the other')
ax.set_title(f"A. Closeness in visual experience\n(Mantel ρ = {ref['rho']:.3f}, p = {ref['p']:.3f})", fontsize=10.5, loc='left')
ax.legend(frameon=False, fontsize=8, loc='upper right'); clean(ax)
# occipital course
r = json.load(open(res('23_three_groups_cases_connectivity', 'connectivity_res.json')))['occipital_second']
CLV = {'P01', 'P15', 'P02', 'P13', 'P17', 'P06', 'P03', 'P10', 'P23'}; LVL = {'P18', 'P21', 'P16', 'P20'}
G = lambda n: 'Congenitally limited vision' if n in CLV else ('Later vision loss' if n in LVL else ('Sighted' if GR.get(n) == 'Sighted' else None))
axs = [fig.add_subplot(gs[1])]; axs.append(fig.add_subplot(gs[2], sharey=axs[0]))
for j, (bx, (c, t)) in enumerate(zip(axs, [('B2_touch', 'B. Occipital alpha while touching\n(ordered trend p = 0.003)'), ('B2_img', 'C. Occipital alpha while imagining\n(ordered trend p = 0.011)')])):
    x = np.arange(7) + 0.5
    for g in ['Sighted', 'Later vision loss', 'Congenitally limited vision']:
        M = np.array([r[c][n][:7] for n in r[c] if G(n) == g]); m = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
        bx.plot(x, m, '-o', color=GCOL.get(g, GCOL['Congenitally low vision']), lw=2, ms=6, markeredgecolor='white', markeredgewidth=1.2, label=f'{g} (n = {len(M)})')
        bx.fill_between(x, m - se, m + se, color=GCOL.get(g, GCOL['Congenitally low vision']), alpha=0.15, lw=0)
    bx.set_title(t, fontsize=10.5, loc='left'); bx.set_xlabel('time (s)'); bx.set_xticks(range(8)); clean(bx)
axs[0].set_ylabel('occipital alpha minus whole-head mean (log)\nlower = occipital cortex more engaged')
plt.setp(axs[1].get_yticklabels(), visible=False)
axs[1].legend(frameon=False, fontsize=8.5, loc='upper center', bbox_to_anchor=(-0.1, -0.16), ncol=3)
save(fig, 'Figure8_experience_and_occipital')
