import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "results", _os.path.basename(_os.path.dirname(_os.path.abspath(__file__)))))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
"""Shapes predicted from EEG through the superquadric bridge (stage 09), as measured and with differences enlarged."""
import json, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '../../src/09_superquadric_bridge'); from sq import PAR, draw
R = json.load(open('sq_res.json')); CL = ['cone', 'cube', 'sphere', 'pyramid', 'cylinder']; W = list(R)
def figure(scale, fname, title):
    fig = plt.figure(figsize=(12.5, 10))
    for i, c in enumerate(CL):
        ax = fig.add_subplot(5, len(W) + 1, i * (len(W) + 1) + 1, projection='3d'); draw(ax, PAR[c], ('TRUE SHAPE\n' if i == 0 else '') + c)
        for j, w in enumerate(W):
            m = {k: np.array(v) for k, v in R[w]['mean'].items()}; centre = np.mean(list(m.values()), 0)
            p = centre + scale * (m[c] - centre)
            ok = np.linalg.norm(m[c] - np.array(PAR[c])) < np.linalg.norm(centre - np.array(PAR[c]))
            head = (w.replace(' · ', '\n') + f"\n{R[w]['acc'] * 100:.0f}% (p={R[w]['p']:.2f})\n") if i == 0 else ''
            draw(fig.add_subplot(5, len(W) + 1, i * (len(W) + 1) + j + 2, projection='3d'), p, head + ('✓ right direction' if ok else '× wrong direction'))
    fig.suptitle(title, fontsize=10); plt.savefig(fname, dpi=90, bbox_inches='tight'); plt.close(fig)
figure(1, 'eeg_shapes.png', 'Mean shape predicted from EEG for each true shape (participants never seen by the model).\n"✓ right direction": the prediction for that shape moved from the average towards the true shape')
figure(8, 'eeg_shapes_enlarged.png', 'Same predictions, differences between shapes enlarged 8 times (for visualisation only).\n"✓ right direction": the prediction for that shape moved from the average towards the true shape')
