import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '06_shape_decoding_main'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, pickle, json, sys, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
D = pickle.load(open('../../data/derived/06_features/feats.pkl', 'rb'))
CL = ['cone', 'cube', 'sphere', 'pyramid', 'cylinder']
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 500
EXCL = set(sys.argv[2].split(',')) if len(sys.argv) > 2 and sys.argv[2] else set()
TAG = sys.argv[3] if len(sys.argv) > 3 else 'main'

def z(X): return (X - X.mean(0)) / (X.std(0) + 1e-9)
# position→shape sequences per block (missing positions from other people in the same block)
def seqs(cond):
    S = {}
    for n, d in D.items():
        if cond in d: S[n] = dict(zip(d[cond]['pos'].tolist(), d[cond]['y'].tolist()))
    common = {}
    for n, m in S.items():
        if n == 'P23' and cond.startswith('B2'): continue
        for p, l in m.items(): common.setdefault(p, l)
    for n in S:
        base = dict(common) if not (n == 'P23' and cond.startswith('B2')) else {15: 'pyramid'}
        base.update(S[n]); S[n] = base
    return S
SEQ = {c: seqs(c) for c in ['B1_erp', 'B1_img', 'B2_touch', 'B2_img', 'B3_erp', 'B3_perc', 'B3_img']}

def labels(n, cond, perm=None):
    pos = D[n][cond]['pos']
    if perm is None: return D[n][cond]['y']
    return np.array([SEQ[cond][n][perm[p - 1]] for p in pos])

def clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))

def run(train_c, test_c, perm=None, names=None):
    names = names or [n for n in D if train_c in D[n] and test_c in D[n] and n not in EXCL]
    pool = [n for n in D if train_c in D[n] and n not in EXCL]
    acc = {}
    for n in names:
        tr = [m for m in pool if m != n]
        Xtr = np.vstack([z(D[m][train_c]['X']) for m in tr]); ytr = np.concatenate([labels(m, train_c, perm) for m in tr])
        c = clf().fit(Xtr, ytr)
        acc[n] = float(c.score(z(D[n][test_c]['X']), D[n][test_c]['y']))
    return acc

TESTS = [('B1_erp', 'B1_erp'), ('B1_img', 'B1_img'), ('B2_touch', 'B2_touch'), ('B2_img', 'B2_img'),
         ('B3_erp', 'B3_erp'), ('B3_perc', 'B3_perc'), ('B3_img', 'B3_img'),
         ('B2_touch', 'B2_img'), ('B3_perc', 'B3_img'), ('B1_img', 'B2_img'), ('B2_img', 'B1_img'), ('B3_perc', 'B2_touch')]
rng = np.random.default_rng(2026)
PERMS = [rng.permutation(np.arange(1, 16)) for _ in range(NPERM)]
out = {}
for a, b in TESTS:
    real = run(a, b); m = float(np.mean(list(real.values())))
    null = Parallel(n_jobs=2)(delayed(lambda p: float(np.mean(list(run(a, b, p).values()))))(p) for p in PERMS)
    null = np.array(null); p = float((np.sum(null >= m) + 1) / (NPERM + 1))
    out[f'{a}->{b}'] = dict(acc=m, per_subject=real, null_mean=float(null.mean()), null95=float(np.percentile(null, 95)), p=p, n=len(real))
    print(f'{a:7s}->{b:7s} n={len(real):2d} accuracy={m:.3f}  chance mean={null.mean():.3f} 95%={np.percentile(null,95):.3f}  p={p:.3f}', flush=True)
# Holm (11 primary: the two directions of B1_img<->B2_img count as one test -> their mean)
json.dump(out, open(f'../../results/06_shape_decoding_main/res_{TAG}.json', 'w'), ensure_ascii=False)
