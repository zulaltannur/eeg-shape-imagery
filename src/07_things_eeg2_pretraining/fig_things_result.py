import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "results", _os.path.basename(_os.path.dirname(_os.path.abspath(__file__)))))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
"""THINGS-EEG2-pretrained encoder + leave-one-subject-out fine-tuning (stage 07)."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open('apply_res.json'))
NAME = {'B1_word': 'Block 1 · word onset', 'B1_img': 'Block 1 · imagery', 'B2_touch': 'Block 2 · touch', 'B2_img': 'Block 2 · imagery',
        'B3_picture': 'Block 3 · picture onset', 'B3_viewing': 'Block 3 · picture viewing', 'B3_img': 'Block 3 · imagery'}
keys = [k for k in R if k.startswith('finetuned|')]
def label(k):
    a, b = k.split('|')[1].split('->')
    return NAME[a] if a == b else f"Transfer · {NAME[a].split(' · ')[1]} → {NAME[b]}"
fig, ax = plt.subplots(figsize=(10, 6.5)); y = np.arange(len(keys))[::-1]
for yy, k in zip(y, keys):
    r = R[k]; acc = r['acc'] * 100
    ax.barh(yy, acc, color='#f48fb1' if r['p'] < 0.05 else '#9e9e9e', height=0.6)
    ax.plot([r['null95'] * 100] * 2, [yy - 0.35, yy + 0.35], color='k', lw=2)
    ax.text(acc + 0.5, yy, f"{acc:.1f}%  p={r['p']:.3f}", va='center', fontsize=9)
ax.set_yticks(y, [label(k) for k in keys], fontsize=9); ax.axvline(20, color='#555', ls='--'); ax.set_xlim(0, 40)
ax.set_xlabel('Cross-subject accuracy (%) — dashed: chance 20%, black: chance 95% bound (uncorrected)')
ax.set_title('Encoder pretrained on THINGS-EEG2 + leave-one-subject-out fine-tuning')
plt.tight_layout(); plt.savefig('things_result.png', dpi=100)
