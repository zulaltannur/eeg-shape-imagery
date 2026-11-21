"""Exploratory analyses (outside the pre-registered plan; labelled separately, BH-FDR correction applied)."""
import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '06_shape_decoding_main'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, pickle, json, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import spearmanr
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
RI = pickle.load(open('../../data/derived/06_features/feats_riem.pkl', 'rb')); OL = pickle.load(open('../../data/derived/06_features/feats.pkl', 'rb'))
CURVED = {'sphere', 'cylinder', 'cone'}; POINTED = {'cone', 'pyramid'}
TARGET = {'identity': lambda y: y, 'curved-cornered': lambda y: np.array(['curved' if s in CURVED else 'cornered' for s in y]),
          'pointed': lambda y: np.array(['pointed' if s in POINTED else 'not' for s in y])}
def z(X): return (X - X.mean(0)) / (X.std(0) + 1e-9)
def seq(D, cond):
    S = {n: dict(zip(d[cond]['pos'].tolist(), d[cond]['y'].tolist())) for n, d in D.items() if cond in d}
    common = {}
    for n, m in S.items():
        if not (n == 'P23' and cond.startswith('B2')): [common.setdefault(p, l) for p, l in m.items()]
    for n in S:
        b = dict(common) if not (n == 'P23' and cond.startswith('B2')) else {15: 'pyramid'}; b.update(S[n]); S[n] = b
    return S
def lab(D, S, n, cond, perm):
    if perm is None: return D[n][cond]['y']
    return np.array([S[n][perm[p - 1]] for p in D[n][cond]['pos']])
def run(D, a, b, tgt, perm=None, S=None):
    f = TARGET[tgt]; names = [n for n in D if a in D[n] and b in D[n]]; pool = [n for n in D if a in D[n]]; acc = []
    for n in names:
        tr = [m for m in pool if m != n]
        c = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, class_weight='balanced'))
        c.fit(np.vstack([z(D[m][a]['X']) for m in tr]), f(np.concatenate([lab(D, S, m, a, perm) for m in tr])))
        yt = f(D[n][b]['y']); pr = c.predict(z(D[n][b]['X']))
        acc.append(np.mean(pr == yt) if tgt == 'identity' else balanced_accuracy_score(yt, pr))
    return float(np.mean(acc)), len(names)
def rsa(D, cond, perm=None, S=None):
    CL = ['cone', 'cube', 'sphere', 'pyramid', 'cylinder']; R = []
    for n in D:
        if cond not in D[n]: continue
        X = z(D[n][cond]['X']); y = lab(D, S, n, cond, perm)
        M = np.array([X[y == c].mean(0) for c in CL])
        if np.isnan(M).any(): continue
        R.append(1 - np.corrcoef(M)[np.triu_indices(5, 1)])
    R = np.array(R)
    return float(np.mean([spearmanr(R[i], np.delete(R, i, 0).mean(0)).correlation for i in range(len(R))]))
rng = np.random.default_rng(99); PERMS = [rng.permutation(np.arange(1, 16)) for _ in range(200)]
jobs = []
for c in ['B1_img', 'B2_touch', 'B2_img', 'B3_perc', 'B3_img']:
    for t in TARGET: jobs.append(('Riemann', RI, c, c, t))
for a, b in [('B2_touch', 'B2_img'), ('B3_perc', 'B3_img')]:
    for t in TARGET: jobs.append(('Riemann', RI, a, b, t))
for c in ['B1_erp', 'B3_erp']:
    for t in ['curved-cornered', 'pointed']: jobs.append(('ERP', OL, c, c, t))
out = []
for fs, D, a, b, t in jobs:
    S = seq(D, a); r, n = run(D, a, b, t)
    nl = np.array(Parallel(n_jobs=2)(delayed(lambda p: run(D, a, b, t, p, S)[0])(p) for p in PERMS))
    p = (np.sum(nl >= r) + 1) / 201
    out.append(dict(feature=fs, training=a, test=b, target=t, n=n, accuracy=r, chance_mean=float(nl.mean()), chance95=float(np.percentile(nl, 95)), p=float(p)))
    print(f'{fs:8s} {a}->{b} {t:12s} n={n} accuracy={r:.3f} chance={nl.mean():.3f}/{np.percentile(nl,95):.3f} p={p:.3f}', flush=True)
for c in ['B1_img', 'B2_touch', 'B2_img', 'B3_perc', 'B3_img']:
    S = seq(RI, c); r = rsa(RI, c); nl = np.array([rsa(RI, c, p, S) for p in PERMS]); p = (np.sum(nl >= r) + 1) / 201
    out.append(dict(feature='Riemann-RSA', training=c, test=c, target='similarity consistency', n=0, accuracy=r, chance_mean=float(nl.mean()), chance95=float(np.percentile(nl, 95)), p=float(p)))
    print(f'RSA {c} consistency r={r:.3f} chance={nl.mean():.3f}/{np.percentile(nl,95):.3f} p={p:.3f}', flush=True)
# BH-FDR
ps = np.array([o['p'] for o in out]); o_ = np.argsort(ps); m = len(ps); q = np.empty(m); prev = 1
for k, i in enumerate(o_[::-1]):
    rank = m - k; prev = min(prev, ps[i] * m / rank); q[i] = prev
for o, qq in zip(out, q): o['q_FDR'] = float(qq)
json.dump(out, open('../../results/06_shape_decoding_main/res_exploration.json', 'w'), ensure_ascii=False)
