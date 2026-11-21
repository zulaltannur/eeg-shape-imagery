"""Supplementary S3: per-person shape tables with anonymous labels (P = English, K = Turkish captions use the same number).
Same calibration and drawing as stage 17; one file per participant."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import sys, json, numpy as np
from style import *
sys.path.insert(0, os.path.join(ROOT, 'src', '09_superquadric_bridge')); from sq import PAR, draw
PR = np.array([PAR[c] for c in CL]); R = json.load(open(res('16_accuracy_boosting', 'final_res.json')))
EXCLUDE = set()
ROWS = [('Block 1 · word onset', 'Block 1\nword onset'), ('Block 1 · imagery', 'Block 1\nimagery'), ('Block 2 · touch', 'Block 2\ntouch'),
        ('Block 2 · imagery', 'Block 2\nimagery'), ('Block 2 · touch to imagery', 'Block 2\ntouch to imagery'),
        ('Block 3 · picture onset', 'Block 3\npicture onset'), ('Block 3 · picture viewing', 'Block 3\npicture viewing'), ('Block 3 · imagery', 'Block 3\nimagery')]
def soft(P, T): L = np.log(P + 1e-9) / T; E = np.exp(L - L.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)
def cal(w, n):
    d = R[w]; P = np.vstack([d[m]['P'] for m in d if m != n]); Y = np.concatenate([[CL.index(y) for y in d[m]['y']] for m in d if m != n])
    Ts = np.linspace(0.05, 1.5, 59); T = Ts[int(np.argmax([np.log(soft(P, T)[np.arange(len(Y)), Y] + 1e-12).mean() for T in Ts]))]
    return soft(np.array(d[n]['P']), T)
os.makedirs(os.path.join(OUT, 'S3'), exist_ok=True)
import pandas as pd
_k = pd.read_excel(os.path.join(ROOT, 'participants', 'participant_information.xlsx'))
GRP = dict(zip(_k['code'], _k['analysis_group'])); AGE = dict(zip(_k['code'], _k['age']))
LOSS_YEARS = {'P20': 22, 'P18': 15, 'P21': 38, 'P16': 14}   # age at complete loss of typical vision (same values as stage 21 mantel.py)
def background(n):
    g = GRP[n]; a = AGE[n]
    if n == 'P23': return f'Later vision loss · age {a} · limited vision from birth for 10 years, then loss; blind for {a - 10} years'
    if g == 'Sighted': return f'Sighted · age {a} · {a} years of typical vision'
    if g == 'Later vision loss':
        L = LOSS_YEARS[n]; return f'Later vision loss · age {a} · {L} years of typical vision, loss {a - L} years ago'
    return f'{g} · age {a} · visual experience limited since birth'
people = sorted({n for d in R.values() for n in d} - EXCLUDE, key=lambda n: anon(n))
for n in people:
    rows = []
    for w, lab in ROWS:
        if n not in R[w]: continue
        Pc = cal(w, n); y = np.array(R[w][n]['y']); correct = np.array(CL)[Pc.argmax(1)] == y; cell = []
        for c in CL:
            idx = np.where(y == c)[0]
            if not len(idx): cell.append(None); continue
            pr = Pc[idx].mean(0); cell.append(((Pc[idx] @ PR).mean(0), f'{CL[int(pr.argmax())]} {pr.max() * 100:.0f}%\n{int(correct[idx].sum())}/{len(idx)} correct'))
        rows.append((lab, correct.mean(), cell))
    nr = len(rows) + 1; fig = plt.figure(figsize=(12, 2.15 * nr + 0.6))
    ax = fig.add_subplot(nr, 6, 1); ax.axis('off'); ax.text(0.5, 0.5, 'Shape presented', ha='center', va='center', fontsize=10, weight='bold')
    for j, c in enumerate(CL): draw(fig.add_subplot(nr, 6, j + 2, projection='3d'), PAR[c], c)
    for i, (lab, acc, cell) in enumerate(rows):
        ax = fig.add_subplot(nr, 6, (i + 1) * 6 + 1); ax.axis('off'); ax.text(0.5, 0.5, lab + f'\naccuracy {acc * 100:.0f}%', ha='center', va='center', fontsize=9)
        for j, h in enumerate(cell):
            a = fig.add_subplot(nr, 6, (i + 1) * 6 + j + 2, projection='3d')
            if h is None: a.set_axis_off(); continue
            draw(a, h[0], h[1])
    fig.suptitle(f'{anon(n)} · {background(n)}\nShapes generated from EEG (the model never saw this person; mean of trials per shape)', fontsize=11)
    fig.savefig(os.path.join(OUT, 'S3', f'FigureS3_{anon(n)}.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
print(len(people), 'tables')
