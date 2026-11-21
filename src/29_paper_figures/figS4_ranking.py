"""Supplementary S4: person ranking map (anonymous labels). Each row label gives the participant's age and visual history;
the two background columns (years of typical vision, residual vision level) are not ranked as measures."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, pandas as pd
from style import *
from matplotlib.colors import LinearSegmentedColormap
Rk = pd.read_csv(res('26_sound_and_ranking', 'ranks.csv'), index_col=0)
Rk = Rk[[c for c in Rk.columns if not c.startswith('Person')]]
Rk.columns = [c.replace('Readability', 'Decodability').replace('readability', 'decodability').replace('congenitally limited', 'congenitally limited vision') for c in Rk.columns]
CLV = {'P01', 'P15', 'P02', 'P13', 'P17', 'P06', 'P03', 'P10', 'P23'}; LVL = {'P18', 'P21', 'P16', 'P20'}
grp = lambda n: 0 if n in CLV else (1 if n in LVL else 2)
rows = sorted(Rk.index, key=lambda n: (grp(n), anon(n))); Rk = Rk.loc[rows]
_k = pd.read_excel(os.path.join(ROOT, 'participants', 'participant_information.xlsx'))
GRP = dict(zip(_k['code'], _k['analysis_group'])); AGE = dict(zip(_k['code'], _k['age']))
LOSS_YEARS = {'P20': 22, 'P18': 15, 'P21': 38, 'P16': 14}
def history(n):
    a = AGE[n]
    if n == 'P23': return f'age {a}: limited vision 10 y, then loss; blind {a - 10} y'
    if GRP[n] == 'Sighted': return f'age {a}: typical vision {a} y'
    if GRP[n] == 'Later vision loss': L = LOSS_YEARS[n]; return f'age {a}: typical vision {L} y, loss {a - L} y ago'
    return f'age {a}: {"blind" if GRP[n] == "Congenitally blind" else "low vision"} since birth'
N = Rk.notna().sum(); P = 1 - (Rk - 1) / (N - 1)
fig, ax = plt.subplots(figsize=(0.42 * Rk.shape[1] + 4, 0.42 * len(rows) + 5.5))
im = ax.imshow(P.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
for i in range(len(rows)):
    for j in range(Rk.shape[1]):
        if not np.isnan(Rk.values[i, j]): ax.text(j, i, int(Rk.values[i, j]), ha='center', va='center', fontsize=6.5, color='white' if P.values[i, j] > 0.6 else INK)
ax.set_xticks(range(Rk.shape[1]), Rk.columns, rotation=60, ha='right', fontsize=7.5); ax.set_yticks(range(len(rows)), [f'{anon(n)} · {history(n)}' for n in rows], fontsize=8)
for y in [i for i in range(1, len(rows)) if grp(rows[i]) != grp(rows[i - 1])]: ax.axhline(y - 0.5, color=INK, lw=1.2)
for i, g in enumerate(['Congenitally limited vision', 'Later vision loss', 'Sighted']):
    idx = [k for k, n in enumerate(rows) if grp(n) == i]
    ax.annotate(g, xy=(-0.27, 1 - (np.mean(idx) + 0.5) / len(rows)), xycoords='axes fraction', ha='center', va='center', fontsize=8.5, rotation=90, weight='bold')
cb = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.01); cb.set_label('darker = higher rank (number = rank)', fontsize=8)
ax.set_title('Each participant\'s rank on each measure (1 = highest value; empty = no recording in that block)', fontsize=11, loc='left')
save(fig, 'FigureS4_ranking_map')
