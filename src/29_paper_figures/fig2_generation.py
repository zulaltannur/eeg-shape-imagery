"""Figure 2: shapes generated from EEG in never-seen people, final system, four main conditions."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import sys, json, numpy as np
from style import *
sys.path.insert(0, os.path.join(ROOT, 'src', '09_superquadric_bridge')); from sq import PAR, surface
from matplotlib.colors import LightSource
r = json.load(open(res('16_accuracy_boosting', 'final_res.json')))
sig = json.load(open(res('16_accuracy_boosting', 'final_res_sig.json')))
conds = ['Block 1 · imagery', 'Block 2 · touch to imagery', 'Block 3 · picture onset', 'Block 3 · imagery']
LS = LightSource(azdeg=315, altdeg=40); SURF = {c: surface(*PAR[c], n=32) for c in CL}
fig = plt.figure(figsize=(13, 13)); outer = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.16)
for b, name in enumerate(conds):
    d = r[name]; Y = np.concatenate([d[n]['y'] for n in d]); P = np.concatenate([np.array(d[n]['P']) for n in d])
    Y = np.array(Y); Yl = np.array(CL)[Y] if Y.dtype.kind in 'iu' else Y
    Pd = np.array(CL)[P.argmax(1)]
    M = np.array([[np.mean(Pd[Yl == t] == g) for g in CL] for t in CL]); acc = np.mean(Pd == Yl)
    inner = outer[b // 2, b % 2].subgridspec(6, 6, hspace=0.02, wspace=0.02)
    for i, t in enumerate(CL):
        ax = fig.add_subplot(inner[i + 1, 0]); ax.axis('off'); ax.text(0.95, 0.5, t, ha='right', va='center', fontsize=10, color=INK)
        for j, g in enumerate(CL):
            if i == 0:
                a2 = fig.add_subplot(inner[0, j + 1]); a2.axis('off'); a2.text(0.5, 0.1, g, ha='center', va='bottom', fontsize=10, color=INK)
            ax = fig.add_subplot(inner[i + 1, j + 1], projection='3d'); X, Yy, Z = SURF[g]
            al = float(np.clip(M[i, j] / 0.45, 0.06, 1)); col = (0.72, 0.12, 0.10) if i == j else (0.30, 0.30, 0.30)
            ax.plot_surface(X, Yy, Z, color=col, alpha=al, shade=True, lightsource=LS, linewidth=0, antialiased=False)
            ax.set_box_aspect((1, 1, 1)); L = 0.6; ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L); ax.view_init(18, 35); ax.set_axis_off(); ax.set_facecolor((1, 1, 1, 0)); ax.patch.set_alpha(0)
            ax.text2D(0.5, -0.04, f'{M[i, j] * 100:.0f}%', transform=ax.transAxes, ha='center', fontsize=8, color='#b71c1c' if i == j else MUT, clip_on=False, zorder=10)
    ax = fig.add_subplot(outer[b // 2, b % 2]); ax.axis('off')
    s = sig[name]; assert abs(s['acc'] - acc) < 0.006, (name, s['acc'], acc)
    ax.set_title(f"{'ABCD'[b]}. {name.replace(' · ', ', ')}\ncorrect shape: {s['acc'] * 100:.1f}% (chance 20%, p = {s['p']:.3f})", fontsize=11, loc='left')
fig.text(0.06, 0.5, 'Shape presented (rows)', rotation=90, va='center', fontsize=11, color=INK)
fig.text(0.5, 0.955, 'Shape generated from EEG (columns)', ha='center', fontsize=11, color=INK)
save(fig, 'Figure2_generation_matrix')
