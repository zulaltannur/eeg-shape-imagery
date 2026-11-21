"""Figure 3: confusion matrices. Figure 4: accuracy by vision group (A) and training source for congenitally low vision (B)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json, numpy as np
from style import *
r = json.load(open(res('18_detailed_analyses', 'detail_res.json')))
# ---- Figure 3
ws = ['Block 1 · imagery', 'Block 2 · touch to imagery', 'Block 3 · picture onset', 'Block 3 · imagery']
fig, ax = plt.subplots(1, 4, figsize=(17, 4.9))
for k, (a, w) in enumerate(zip(ax, ws)):
    A = r['A_confusion_geometry'][w]; M = np.array(A['confusion'])
    im = a.imshow(M * 100, cmap='Blues', vmin=14, vmax=30)
    for i in range(5):
        for j in range(5):
            a.text(j, i, f'{M[i, j] * 100:.0f}', ha='center', va='center', fontsize=8.5, color='white' if M[i, j] > 0.26 else INK)
    a.set_xticks(range(5), CL, rotation=40, ha='right'); a.set_yticks(range(5), CL)
    a.set_xlabel('shape predicted by the model')
    if k == 0: a.set_ylabel('shape presented')
    st = lambda p: ' *' if p < .05 else ''
    a.set_title(f"{'ABCD'[k]}. {w.replace(' · ', ', ')}\ncurved vs cornered: {A['curved_accuracy'] * 100:.0f}% (p = {A['curved_p']:.3f}){st(A['curved_p'])}\n"
                f"pointed vs not pointed: {A['pointed_accuracy'] * 100:.0f}% (p = {A['pointed_p']:.3f}){st(A['pointed_p'])}", fontsize=9, loc='left')
    for s in a.spines.values(): s.set_visible(False)
plt.subplots_adjust(wspace=0.5); fig.colorbar(im, ax=ax, shrink=0.75, label='mean model probability (%)')
save(fig, 'Figure3_confusion_matrices')
# ---- Figure 4
ws = ['Block 1 · imagery', 'Block 2 · imagery', 'Block 2 · touch to imagery', 'Block 3 · picture onset', 'Block 3 · imagery']
fig = plt.figure(figsize=(14, 9.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.62, wspace=0.08)
ax = fig.add_subplot(gs[0, :]); x = np.arange(len(ws)); bw = 0.2
for k, g in enumerate(GROUPS):
    for i, w in enumerate(ws):
        d = r['C_groups'][w]['groups'].get(g)
        if not d: continue
        xx = x[i] + (k - 1.5) * bw
        ax.bar(xx, d['accuracy'] * 100, bw * 0.9, color=GCOL[g], label=g if i == 0 else None)
        ax.text(xx, d['accuracy'] * 100 + 0.6, f"{d['accuracy'] * 100:.0f}{'*' if d['p_chance'] < .05 else ''}\nn={d['n']}", ha='center', fontsize=7.5, color=INK)
ax.axhline(20, color=MUT, ls='--', lw=1); ax.text(len(ws) - 0.45, 20.6, 'chance 20%', color=MUT, fontsize=8, ha='right')
ax.set_xticks(x, [w.replace(' · ', '\n') + f"\ngroup difference p = {r['C_groups'][w]['group_diff_p']:.3f}" for w in ws])
ax.set_ylabel('accuracy (%)'); ax.set_ylim(0, 42); clean(ax)
ax.legend(frameon=False, ncol=4, loc='upper left', fontsize=8.5)
ax.set_title('A. Accuracy by vision group (* = above chance within the group)', fontsize=11, loc='left')
c19 = json.load(open(res('19_congenital_low_vision_representation', 'clv_res.json')))['A_training_source']
pi = json.load(open(res('19_congenital_low_vision_representation', 'perm_B2_img.json')))
pt = json.load(open(res('19_congenital_low_vision_representation', 'perm_B2_touch.json')))
def pv(s): nl = np.array(s['null']); return (1 + np.sum(nl >= s['obs'])) / (1 + len(nl))
src = ['Congenitally low vision', 'Later vision loss', 'Sighted']
for j, (cn, st, ttl) in enumerate([('B2 touch', pt, 'Block 2, touch'), ('B2 imagery', pi, 'Block 2, imagery after touch')]):
    a = fig.add_subplot(gs[1, j], sharey=None if j == 0 else fig.axes[-1]); c = c19[cn]['main']; xx = 0; ticks = []; labs = []
    for test in ['Congenitally low vision', 'Sighted']:
        for s in src:
            key = f'{test} ← {s}'
            if key not in c: continue
            d = c[key]; a.bar(xx, d['accuracy'] * 100, 0.75, color=GCOL[s], label=s if test == 'Congenitally low vision' else None)
            v = np.array(list(d['person'].values())) * 100
            a.scatter(xx + np.linspace(-0.2, 0.2, len(v)), v, s=11, color=INK, alpha=0.45, zorder=3)
            a.text(xx, d['accuracy'] * 100 + 1.2, f"{d['accuracy'] * 100:.0f}{'*' if d['p'] < .05 else ''}", ha='center', fontsize=8.5, color=INK); xx += 1
        ticks.append(xx - 2); labs.append(f"tested on: {test.lower()}\n(n = {len(c[test + ' ← Sighted']['person'])})"); xx += 0.8
    a.axhline(20, color=MUT, ls='--', lw=1); a.set_xticks(ticks, labs); a.set_ylim(0, 65); clean(a)
    a.set_title(f"{'BC'[j]}. {ttl}\nown-group training minus sighted training: {st['obs'] * 100:+.1f} points, p = {pv(st):.3f}", fontsize=10, loc='left')
    if j == 0:
        a.set_ylabel('accuracy (%), dots = individual people')
        a.legend(title='group the model was trained on', frameon=False, fontsize=8, title_fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 1.0))
    else:
        plt.setp(a.get_yticklabels(), visible=False)
save(fig, 'Figure4_vision_groups')
