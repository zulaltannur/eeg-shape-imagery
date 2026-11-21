import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "results", _os.path.basename(_os.path.dirname(_os.path.abspath(__file__)))))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
"""Pre-registered shape decoding (stage 06): leave-one-subject-out accuracy and accuracy over time."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open('res_main.json')); S = json.load(open('res_secondary.json'))
rows = [('B1_erp->B1_erp', 'Block 1 · response to the word'), ('B1_img->B1_img', 'Block 1 · imagery'), ('B2_touch->B2_touch', 'Block 2 · touch'),
        ('B2_img->B2_img', 'Block 2 · imagery'), ('B3_erp->B3_erp', 'Block 3 · response to the picture'), ('B3_perc->B3_perc', 'Block 3 · picture viewing (rhythm)'),
        ('B3_img->B3_img', 'Block 3 · imagery'), ('B2_touch->B2_img', 'Transfer · touch → imagery'), ('B3_perc->B3_img', 'Transfer · vision → imagery'),
        ('B1_img->B2_img', 'Transfer · word imagery → touch imagery'), ('B2_img->B1_img', 'Transfer · touch imagery → word imagery'), ('B3_perc->B2_touch', 'Transfer · vision → touch')]
# Holm over the 11 primary tests (the two directions of B1_img<->B2_img count as one test, using their mean)
ps = {k: R[k]['p'] for k, _ in rows if k not in ('B1_img->B2_img', 'B2_img->B1_img')}
ps['B1_img<->B2_img'] = (R['B1_img->B2_img']['p'] + R['B2_img->B1_img']['p']) / 2
order = sorted(ps, key=ps.get); holm = {}; run = 0
for i, k in enumerate(order):
    run = max(run, min(1, ps[k] * (len(order) - i))); holm[k] = run
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.6, 1]})
y = np.arange(len(rows))[::-1]
for yy, (k, lab) in zip(y, rows):
    r = R[k]; acc = r['acc'] * 100; ok = holm.get(k, 1) < 0.05
    a.barh(yy, acc, color='#c2185b' if ok else ('#f48fb1' if r['p'] < 0.05 else '#9e9e9e'), height=0.6)
    a.plot([r['null95'] * 100] * 2, [yy - 0.35, yy + 0.35], color='k', lw=2)
    a.text(acc + 0.5, yy, f"{acc:.1f}%  p={r['p']:.3f}" + (' (Holm ✓)' if ok else ''), va='center', fontsize=9)
a.set_yticks(y, [f"{lab} (n={R[k]['n']})" for k, lab in rows]); a.axvline(20, color='#555', ls='--'); a.set_xlim(0, 40)
a.set_xlabel('Cross-subject accuracy (%) — dashed line: chance 20%, black line: 95th percentile of the chance distribution')
a.set_title('Shape discrimination: leave one subject out')
for key, col, lab in [('time_B3_erp', '#c2185b', 'Block 3 · response to the picture'), ('time_B1_erp', '#5c6bc0', 'Block 1 · response to the word')]:
    d = S[key]; t = np.array(d['t']) * 1000
    b.plot(t, np.array(d['acc']) * 100, 'o-', color=col, label=lab); b.plot(t, np.array(d['null95']) * 100, ':', color=col)
b.axhline(20, color='#555', ls='--'); b.axvline(0, color='k', lw=0.8); b.legend(fontsize=9)
b.set_xlabel('After the stimulus (ms)'); b.set_ylabel('Accuracy (%)'); b.set_title('Over time (100 ms window; dotted: chance 95%)')
plt.tight_layout(); plt.savefig('shape_discrimination.png', dpi=105)
