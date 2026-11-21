"""Supplementary S6: approaches that were tried and did not exceed chance.
A: hidden Markov model of tactile exploration (state signatures, cropped from the stage 13 figure; no participant labels);
B: share of time in each state per participant (anonymous labels); C: direct superquadric regression (stage 09 figure)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json, numpy as np
from PIL import Image
from style import *
ST = ['flat surface', 'curved surface', 'edge', 'corner/tip', 'all hand grasp']
im = Image.open(res('13_tactile_hmm_touchsim', 'hmm_states.png')); left = np.array(im.crop((0, 0, 870, 495)))
sq = np.array(Image.open(res('09_superquadric_bridge', 'eeg_shapes_enlarged.png')))
occ = json.load(open(res('13_tactile_hmm_touchsim', 'hmm_occ.json'))); t = json.load(open(res('13_tactile_hmm_touchsim', 'hmm_test.json')))
names = sorted(occ, key=anon); O = np.array([occ[n] for n in names])
fig = plt.figure(figsize=(15, 15)); gs = fig.add_gridspec(2, 2, height_ratios=[0.42, 1], hspace=0.12, wspace=0.32)
a = fig.add_subplot(gs[0, 0]); a.imshow(left); a.axis('off')
a.set_title(f"A. Hidden Markov model of touch: shared EEG signature of each contact state\n(shape decoding from the state sequence {t['acc'] * 100:.1f}%, p = {t['p']:.3f}; chance 20%)", fontsize=10.5, loc='left')
b = fig.add_subplot(gs[0, 1]); h = b.imshow(O.T, cmap='Reds', aspect='auto')
b.set_yticks(range(5), ST); b.set_xticks(range(len(names)), [anon(n) for n in names], rotation=70, fontsize=8)
plt.colorbar(h, ax=b, label='share of time'); b.set_title('B. Share of time spent in each state, per participant', fontsize=10.5, loc='left')
c = fig.add_subplot(gs[1, :]); c.imshow(sq); c.axis('off')
c.set_title('C. Direct regression of superquadric parameters from EEG (differences between shapes enlarged 8 times for display)', fontsize=10.5, loc='left')
save(fig, 'FigureS6_tried_methods')
