import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "results", _os.path.basename(_os.path.dirname(_os.path.abspath(__file__)))))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
"""Shapes from the CAD coordinates and the two geometry models (stage 08)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
g = np.load('geom.npz', allow_pickle=True); CL = list(g['CL'])
fig = plt.figure(figsize=(16.5, 6.8))
for i, c in enumerate(CL):
    ax = fig.add_subplot(2, 5, i + 1, projection='3d'); p = g[f'pc_{c}'][::2]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 2], cmap='Reds', s=1.5, vmin=p[:, 2].min() - 0.5); ax.set_axis_off(); ax.set_title(c)
    ax.set_box_aspect((1, 1, 1))
for j, (M, title, unit) in enumerate([(g['chamfer'], 'Form similarity\n(Chamfer distance from the coordinates)', 'mm'),
                                       (g['desc_dist'], 'Structure similarity\n(corners, edges, faces, curvature, pointedness, volume, area)', 'z-distance')]):
    ax = fig.add_subplot(2, 2, 3 + j); im = ax.imshow(M, cmap='Reds')
    for a in range(5):
        for b in range(5): ax.text(b, a, f'{M[a, b]:.1f}', ha='center', va='center', fontsize=9, color='white' if M[a, b] > M.max() * 0.6 else 'k')
    ax.set_xticks(range(5), CL); ax.set_yticks(range(5), CL); ax.set_title(title, fontsize=10); fig.colorbar(im, ax=ax, label=unit, shrink=0.9)
plt.tight_layout(); plt.savefig('geometry.png', dpi=100)
