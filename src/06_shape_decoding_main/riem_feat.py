"""Exploration: Riemann-aligned filter-bank covariance features (label-free within-person re-centring)."""
import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '06_shape_decoding_main'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, glob, pickle, warnings
from scipy.signal import butter, sosfiltfilt
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore'); mne.set_log_level('ERROR')
exec(open('../../src/05_participant_profiles/battery.py').read().split('R = {}')[0])  # chunk_masks, G
SF = 256; CLEAN = '../../data'
CH32 = ['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','Fp2','Fz','Cz']
BANDS = [(4, 7), (8, 13), (14, 25), (26, 40)]
SOS = [butter(4, b, btype='band', fs=SF, output='sos') for b in BANDS]

def covs(segs):
    """segs: list (ch,t) -> per band (n,32.32) using only clean 1-s chunks"""
    m = chunk_masks(segs)
    out = [[] for _ in BANDS]; keep = []
    for s, mm in zip(segs, m):
        idx = np.where(mm)[0]
        if len(idx) == 0: keep.append(False); continue
        keep.append(True)
        for bi, sos in enumerate(SOS):
            f = sosfiltfilt(sos, s, axis=1)
            x = np.concatenate([f[:, c*SF:(c+1)*SF] for c in idx], 1)
            out[bi].append(Covariances('oas').fit_transform(x[None])[0])
    return [np.array(o) for o in out], np.array(keep)

def align_ts(C):
    """within-person re-centring (label-free) + tangent space at identity"""
    M = mean_riemann(C); W = invsqrtm(M)
    Ca = np.array([W @ c @ W for c in C])
    return TangentSpace(metric='riemann').fit_transform(Ca) if False else ts(Ca)
def ts(Ca):
    from pyriemann.utils.tangentspace import tangent_space
    return tangent_space(Ca, np.eye(Ca.shape[1]), metric='riemann')

R = {}
def add(n, cond, segs, y, pos):
    if n == 'P23' and cond.startswith('B2'):
        k = [i for i, p in enumerate(pos) if p != 15]; segs = [segs[i] for i in k]; y = [y[i] for i in k]; pos = [pos[i] for i in k]
    C, keep = covs(segs)
    F = np.hstack([align_ts(c) for c in C])
    R.setdefault(n, {})[cond] = dict(X=F, y=np.array(y)[keep], pos=np.array(pos)[keep])

for n in sorted(G):
    f = glob.glob(f'{CLEAN}/clean_epochs/epo3/{n}_*-epo.fif')
    if f:
        ep = mne.read_epochs(f[0], preload=True).set_eeg_reference('average').pick(CH32); md = ep.metadata.reset_index(drop=True); t = ep.times
        add(n, 'B1_img', [x[:, (t >= 0.5) & (t < 4.5)] for x in ep.get_data()], list(md['shape']), list(md.trial))
    f = [x for x in glob.glob(f'{CLEAN}/clean_epochs/epo2/{n}_block2*_touch_motionless-epo.fif') if 'nocube' not in x and 'third' not in x]
    if f:
        for cond, ff in [('B2_touch', f[0]), ('B2_img', f[0].replace('touch', 'imagery'))]:
            ep = mne.read_epochs(ff, preload=True).set_eeg_reference('average').pick(CH32); md = ep.metadata.reset_index(drop=True)
            add(n, cond, list(ep.get_data()), list(md['shape']), list(md.trial))
    f = glob.glob(f'{CLEAN}/clean_epochs/epo/{n}_block3_visual_perception-epo.fif')
    if f:
        ep = mne.read_epochs(f[0], preload=True).set_eeg_reference('average').pick(CH32); md = ep.metadata.reset_index(drop=True); t = ep.times
        add(n, 'B3_perc', [x[:, (t >= 1) & (t < 5)] for x in ep.get_data()], list(md['shape']), list(md.trial))
        ei = mne.read_epochs(f[0].replace('visual_perception', 'imagery'), preload=True).set_eeg_reference('average').pick(CH32); mi = ei.metadata.reset_index(drop=True); t = ei.times
        add(n, 'B3_img', [x[:, (t >= 1) & (t < 9)] for x in ei.get_data()], list(mi['shape']), list(mi.trial))
    print(n, {k: v['X'].shape for k, v in R.get(n, {}).items()}, flush=True)
pickle.dump(R, open('../../data/derived/06_features/feats_riem.pkl', 'wb'))
