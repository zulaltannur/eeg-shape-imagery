"""Figure 1: study design. (A) the five shapes; (B) geometric distance between them (Chamfer, from the CAD models);
(C) trial timeline and analysis windows of each block; participants per block."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import sys, numpy as np
from style import *
from matplotlib.patches import Rectangle, FancyBboxPatch
sys.path.insert(0, os.path.join(ROOT, 'src', '09_superquadric_bridge')); from sq import PAR, draw
GEOM = res('08_coordinate_geometry', 'geom.npz')
g = np.load(GEOM, allow_pickle=True)
fig = plt.figure(figsize=(16, 11.5)); gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.25], width_ratios=[1.6, 1], hspace=0.28, wspace=0.12)
sub = gs[0, 0].subgridspec(1, 5, wspace=0.0)
for i, c in enumerate(CL): draw(fig.add_subplot(sub[0, i], projection='3d'), PAR[c], c)
ax = fig.add_subplot(gs[0, 0]); ax.axis('off')
ax.set_title('A. Five 3D shapes (60 mm objects for touch; one picture each for vision; one spoken name each for hearing)', fontsize=11, loc='left')
bx = fig.add_subplot(gs[0, 1]); M = g['chamfer']; im = bx.imshow(M, cmap='Greys', vmin=0, vmax=M.max() * 1.25)
for a in range(5):
    for b in range(5): bx.text(b, a, f'{M[a, b]:.1f}', ha='center', va='center', fontsize=9, color='white' if M[a, b] > M.max() * 0.75 else INK)
bx.set_xticks(range(5), CL, rotation=30, ha='right'); bx.set_yticks(range(5), CL)
for s in bx.spines.values(): s.set_visible(False)
plt.colorbar(im, ax=bx, shrink=0.85, label='Chamfer distance (mm)')
bx.set_title('B. Geometric distance between the shapes', fontsize=11, loc='left')
# ---- timelines
cx = fig.add_subplot(gs[1, :]); EV = '#8c959f'; PER = '#4a90b8'; IMG = '#c0699f'
lanes = [
    ('Block 1: hearing\nblindfolded, n = 22\n(10 sighted, 6 congenitally low vision,\n5 later vision loss, 1 congenitally blind)',
     [('beep', 0, 1.1), ('word (onset 1.52–2.81 s)', 1.52, 1.79)], [('word response', 1.52, 2.32, PER), ('imagery', 2.02, 6.02, IMG)],
     'next trial 7–14 s after the beep;\nwindows follow the measured word onset\n(shown here for the earliest word)', 7.2),
    ('Block 2: touch\nblindfolded, n = 21\n(10 sighted, 6 congenitally low vision,\n4 later vision loss, 1 congenitally blind)',
     [('beep', 0, 1.1), ('object picked up, explored with both hands, put back (median 11.2 s)', 1.1, 11.2), ('imagery beep', 13, 14.1)],
     [('touch', 1, 9, PER), ('imagery', 17, 24, IMG)], 'next trial 27–29 s after the touch beep;\ntouch window starts 1 s after the\npick-up detected in the EEG', 25),
    ('Block 3: vision\nn = 11\n(9 sighted, 2 congenitally low vision)',
     [('picture on screen', 0, 5)], [('picture response', 0, 0.8, PER), ('viewing', 1, 5, PER), ('imagery', 6, 14, IMG)],
     'next trial 16–18 s after picture onset', 15)]
for k, (lab, evs, wins, note, nxt) in enumerate(lanes):
    y0 = (2 - k) * 3.9
    cx.text(-1.0, y0 + 1.0, lab, ha='right', va='center', fontsize=9.5, color=INK)
    for j, (en, a, b) in enumerate(evs):
        yy = y0 + 1.55
        cx.add_patch(Rectangle((a, yy), b - a, 0.75, color=EV, alpha=0.35 if 'object' in en else 0.85, lw=0))
        cx.text(a + (b - a) / 2 if b - a > 3 else a, yy + 0.95, en, ha='center' if b - a > 3 else 'left', fontsize=8.5, color=INK)
    for wn, a, b, c in wins:
        cx.add_patch(Rectangle((a, y0 + 0.35), b - a, 0.75, color=c, lw=0))
        if b - a >= 2: cx.text(a + (b - a) / 2, y0 + 0.72, wn, ha='center', va='center', fontsize=8.5, color='white')
        else: cx.annotate(wn, xy=(a + (b - a) / 2, y0 + 0.35), xytext=(a + (b - a) / 2 - 0.3, y0 - 0.25), ha='left', va='top', fontsize=8, color=c,
                          arrowprops=dict(arrowstyle='-', color=c, lw=0.8))
    
    cx.text(nxt, y0 + 1.1, note, ha='left', va='center', fontsize=8, color=MUT)
cx.set_xlim(-0.5, 30); cx.set_ylim(-0.9, 11.3); cx.set_yticks([])
cx.set_xlabel('time from trial onset (s)'); cx.set_xticks(range(0, 31, 2))
for s in ['top', 'right', 'left']: cx.spines[s].set_visible(False)
cx.add_patch(Rectangle((22, 10.6), 0.8, 0.4, color=PER, lw=0)); cx.text(23, 10.8, 'perception window', va='center', fontsize=8.5)
cx.add_patch(Rectangle((26.3, 10.6), 0.8, 0.4, color=IMG, lw=0)); cx.text(27.3, 10.8, 'imagery window', va='center', fontsize=8.5)
cx.set_title('C. One trial of each block and the analysis windows (each block: 3 runs × 5 shapes = 15 trials; same order for everyone)', fontsize=11, loc='left')
save(fig, 'Figure1_design')
