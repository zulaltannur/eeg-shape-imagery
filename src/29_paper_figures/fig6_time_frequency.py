"""Figure 6: congenitally low vision, Block 2 imagery after touch: time windows (A) and frequency bands (B)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json, numpy as np
from style import *
C1 = GCOL['Congenitally low vision']; C2 = GCOL['Sighted']
def L(c, s):
    d = json.load(open(res('20_time_frequency', f'res_{c}_{s}.json'))); nl = np.array(d['null'])
    d['p'] = (1 + np.sum(nl >= d['diff'])) / (1 + len(nl)); return d
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.25))
W = ['z0-2', 'z1-3', 'z2-4', 'z3-5', 'z4-6', 'z5-7']; x = np.arange(6)
for key, col, lab in [('same_group', C1, 'model trained on congenitally low vision'), ('sighted', C2, 'model trained on sighted')]:
    ax.plot(x, [L('B2_img', w)[key] * 100 for w in W], '-o', color=col, lw=2, ms=8, label=lab, markeredgecolor='white', markeredgewidth=1.5)
for i, w in enumerate(W):
    d = L('B2_img', w); ax.text(i, 37, f"p = {d['p']:.2f}", ha='center', fontsize=8, color=INK if d['p'] < .05 else MUT)
ax.axhline(20, color=MUT, ls='--', lw=1); ax.text(5.3, 20.5, 'chance', color=MUT, fontsize=8, ha='right')
ax.set_xticks(x, [w[1:].replace('-', '–') + ' s' for w in W]); ax.set_ylim(5, 40); ax.set_ylabel('accuracy in congenitally low vision (%)')
ax.set_xlabel('time window after imagery onset (2 s windows, 1 s step)')
ax.set_title('A. Time course of imagery', fontsize=11, loc='left'); ax.legend(frameon=False, fontsize=8.5, loc='lower left'); clean(ax)
B = ['theta', 'alpha', 'beta', 'gamma']; xb = np.arange(4); bw = 0.38
for k, (key, col) in enumerate([('same_group', C1), ('sighted', C2)]):
    bx.bar(xb + (k - 0.5) * bw, [L('B2_img', b)[key] * 100 for b in B], bw * 0.92, color=col)
for i, b in enumerate(B):
    d = L('B2_img', b); bx.text(i, 36, f"p = {d['p']:.3f}", ha='center', fontsize=8, color=INK if d['p'] < .05 else MUT, weight='bold' if d['p'] < .05 else None)
bx.axhline(20, color=MUT, ls='--', lw=1)
bx.set_xticks(xb, ['theta\n4–7 Hz', 'alpha\n8–13 Hz', 'beta\n14–25 Hz', 'gamma\n26–40 Hz']); bx.set_ylim(0, 40)
bx.set_title('B. Frequency bands (whole imagery window)', fontsize=11, loc='left'); clean(bx)
save(fig, 'Figure6_time_frequency')
