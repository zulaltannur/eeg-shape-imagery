import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '06_shape_decoding_main'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, pandas as pd, glob, os, warnings, pickle
from scipy.signal import butter, sosfiltfilt
warnings.filterwarnings('ignore'); mne.set_log_level('ERROR')
SF = 256
src = open('../../src/05_participant_profiles/battery.py').read().split('R = {}')[0]
exec(src)  # chunk_masks, bandpow definitions, G
FB = {'theta': (4, 7), 'alpha': (8, 13), 'beta': (14, 25), 'gamma': (26, 40)}
CLEAN = '../../data'
CH = None

def power_feats(segs):
    m = chunk_masks(segs)
    F = []
    for s, mm in zip(segs, m):
        idx = np.where(mm)[0]
        if len(idx) == 0: F.append(None); continue
        ps = []
        for c in idx:
            p, fr = mne.time_frequency.psd_array_welch(s[:, c*SF:(c+1)*SF], SF, fmin=2, fmax=40, n_fft=SF, verbose=False)
            ps.append(p)
        p = np.mean(ps, 0)
        F.append(np.concatenate([np.log(p[:, (fr >= lo) & (fr <= hi)].mean(1)) for lo, hi in FB.values()]))
    return F

def erp_feats(ep):
    e = ep.copy().set_eeg_reference('average').filter(1, 20).apply_baseline((-0.2, 0)).crop(-0.2, 0.8)
    X = e.get_data() * 1e6; t = e.times
    ok = np.ptp(X, axis=2).max(1) < 150
    bins = np.arange(0.05, 0.80, 0.05)
    F = np.stack([X[:, :, (t >= a) & (t < a + 0.05)].mean(2) for a in bins], 2).reshape(len(X), -1)
    # time-resolved: 100 ms window, 50 ms step, 0-0.8
    T = np.stack([X[:, :, (t >= a) & (t < a + 0.1)].mean(2) for a in np.arange(-0.1, 0.75, 0.05)], 2)
    return F, T, ok

D = {}
def add(n, cond, F, y, pos, extra=None):
    keep = [i for i, f in enumerate(F) if f is not None and not (n == 'P23' and cond.startswith('B2') and pos[i] == 15)]
    D.setdefault(n, {})[cond] = dict(X=np.array([F[i] for i in keep]), y=np.array([y[i] for i in keep]), pos=np.array([pos[i] for i in keep]))
    if extra is not None: D[n][cond]['T'] = extra[keep]

for n in sorted(G):
    # Block 1
    f = glob.glob(f'{CLEAN}/clean_epochs/epo3/{n}_*-epo.fif')
    if f:
        ep = mne.read_epochs(f[0], preload=True); md = ep.metadata.reset_index(drop=True)
        F, T, ok = erp_feats(ep)
        add(n, 'B1_erp', [F[i] if ok[i] else None for i in range(len(F))], list(md['shape']), list(md.trial), T)
        X = ep.copy().set_eeg_reference('average').get_data(); t = ep.times
        segs = [x[:, (t >= 0.5) & (t < 4.5)] for x in X]
        add(n, 'B1_img', power_feats(segs), list(md['shape']), list(md.trial))
    # Block 2
    f = [x for x in glob.glob(f'{CLEAN}/clean_epochs/epo2/{n}_block2*_touch_motionless-epo.fif') if 'nocube' not in x and 'third' not in x]
    if f:
        for cond, ff in [('B2_touch', f[0]), ('B2_img', f[0].replace('touch', 'imagery'))]:
            ep = mne.read_epochs(ff, preload=True); md = ep.metadata.reset_index(drop=True)
            X = ep.copy().set_eeg_reference('average').get_data()
            add(n, cond, power_feats(list(X)), list(md['shape']), list(md.trial))
    # Block 3
    f = glob.glob(f'{CLEAN}/clean_epochs/epo/{n}_block3_visual_perception-epo.fif')
    if f:
        ep = mne.read_epochs(f[0], preload=True); md = ep.metadata.reset_index(drop=True)
        F, T, ok = erp_feats(ep)
        add(n, 'B3_erp', [F[i] if ok[i] else None for i in range(len(F))], list(md['shape']), list(md.trial), T)
        X = ep.copy().set_eeg_reference('average').get_data(); t = ep.times
        add(n, 'B3_perc', power_feats([x[:, (t >= 1) & (t < 5)] for x in X]), list(md['shape']), list(md.trial))
        ei = mne.read_epochs(f[0].replace('visual_perception', 'imagery'), preload=True); mi = ei.metadata.reset_index(drop=True)
        X = ei.copy().set_eeg_reference('average').get_data(); t = ei.times
        add(n, 'B3_img', power_feats([x[:, (t >= 1) & (t < 9)] for x in X]), list(mi['shape']), list(mi.trial))
    print(n, {k: len(v['y']) for k, v in D.get(n, {}).items()}, flush=True)
pickle.dump(D, open('../../data/derived/06_features/feats.pkl', 'wb'))
