"""Figure 5: shapes imagined after touch (Block 2, alpha) by congenitally low vision participants,
decoded with their own group's pattern vs the sighted pattern. Same computation and seed as stage 23 group_shapes.py."""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
os.chdir(os.path.join(HERE, '..', '..', 'results', '23_three_groups_cases_connectivity'))
code = open('../../src/23_three_groups_cases_connectivity/group_shapes.py').read()
exec(code.split('COLS=[')[0])
os.chdir(os.path.join(HERE, '..', '..', 'results', '23_three_groups_cases_connectivity'))
import sys; sys.path.insert(0, HERE); from style import save, INK, MUT
COLS = [('own group pattern', 'CLV', 'CLV'), ('sighted pattern', 'CLV', 'Sighted')]
P = {lab: {n: predict(n, grp[sg]) for n in grp[tg]} for lab, tg, sg in COLS}
from matplotlib.colors import LightSource
from sq import surface
LS = LightSource(azdeg=315, altdeg=40); SURF = {c: surface(*PAR[c], n=32) for c in CL}
old = json.load(open('group_shapes.json'))
fig = plt.figure(figsize=(13, 6.8)); outer = fig.add_gridspec(1, 2, wspace=0.12)
for b, (lab, tg, sg) in enumerate(COLS):
    Y = np.concatenate([yy[n] for n in P[lab]]); Pd = np.concatenate([P[lab][n].argmax(1) for n in P[lab]])
    M = np.array([[np.mean(Pd[Y == i] == j) for j in range(5)] for i in range(5)]); acc = np.mean([np.mean(P[lab][n].argmax(1) == yy[n]) for n in P[lab]])
    ref = old[['Congenitally limited vision\nwith own group pattern', 'Congenitally limited vision\nwith the sighted pattern'][b]]
    assert abs(ref['overall_accuracy'] - acc) < 0.002
    inner = outer[0, b].subgridspec(6, 6, hspace=0.02, wspace=0.02)
    for i, t in enumerate(CL):
        ax = fig.add_subplot(inner[i + 1, 0]); ax.axis('off'); ax.text(0.95, 0.5, t, ha='right', va='center', fontsize=10, color=INK)
        for j, g in enumerate(CL):
            if i == 0:
                a2 = fig.add_subplot(inner[0, j + 1]); a2.axis('off'); a2.text(0.5, 0.1, g, ha='center', va='bottom', fontsize=10, color=INK)
            ax = fig.add_subplot(inner[i + 1, j + 1], projection='3d'); X, Yy, Z = SURF[g]
            al = float(np.clip(M[i, j] / 0.45, 0.06, 1)); col = (0.72, 0.12, 0.10) if i == j else (0.30, 0.30, 0.30)
            ax.plot_surface(X, Yy, Z, color=col, alpha=al, shade=True, lightsource=LS, linewidth=0, antialiased=False)
            ax.set_box_aspect((1, 1, 1)); L = 0.6; ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L); ax.view_init(18, 35)
            ax.set_axis_off(); ax.patch.set_alpha(0)
            ax.text2D(0.5, -0.04, f'{M[i, j] * 100:.0f}%', transform=ax.transAxes, ha='center', fontsize=8, color='#b71c1c' if i == j else MUT, clip_on=False, zorder=10)
    ax = fig.add_subplot(outer[0, b]); ax.axis('off')
    ax.set_title(f"{'AB'[b]}. Decoded with the {lab}\noverall accuracy {acc * 100:.0f}% (chance 20%)", fontsize=11, loc='left')
fig.text(0.07, 0.45, 'Shape touched (rows)', rotation=90, va='center', fontsize=11, color=INK)
fig.text(0.5, 0.97, 'Congenitally low vision, Block 2 imagery after touch (alpha rhythm): shape decoded from EEG (columns)', ha='center', fontsize=11, color=INK)
save(fig, 'Figure5_group_shapes')
