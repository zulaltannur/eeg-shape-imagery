import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '08_coordinate_geometry'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, pickle, json, warnings
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
g = np.load('geom.npz'); CL = list(g['CL']); Z = g['desc_z']; MOD = {'form(Chamfer)': g['chamfer'], 'structure(feature)': g['desc_dist']}
iu = np.triu_indices(5, 1)
OL = pickle.load(open('../../data/derived/06_features/feats.pkl', 'rb')); EMr = pickle.load(open('../../data/derived/07_things/ours_emb.pkl', 'rb'))
EM = {n: {c: dict(X=v['Z'], y=v['y'], pos=v['pos']) for c, v in d.items()} for n, d in EMr.items()}
SETS = {'main': (OL, ['B1_erp', 'B1_img', 'B2_touch', 'B2_img', 'B3_erp', 'B3_perc', 'B3_img'], [('B2_touch', 'B2_img'), ('B3_perc', 'B3_img')]),
        'THINGS': (EM, ['B1_word', 'B1_img', 'B2_touch', 'B2_img', 'B3_picture', 'B3_viewing', 'B3_img'], [('B2_touch', 'B2_img'), ('B3_viewing', 'B3_img')])}
def z(X): return (X - X.mean(0)) / (X.std(0) + 1e-9)
def seqs(D, c):
    S = {n: dict(zip(D[n][c]['pos'].tolist(), D[n][c]['y'].tolist())) for n in D if c in D[n]}; com = {}
    for n, m in S.items():
        if not (n == 'P23' and c.startswith('B2')): [com.setdefault(p, l) for p, l in m.items()]
    for n in S:
        b = dict(com) if not (n == 'P23' and c.startswith('B2')) else {15: 'pyramid'}; b.update(S[n]); S[n] = b
    return S
def lab(D, S, n, c, perm): return D[n][c]['y'] if perm is None else np.array([S[n][perm[p - 1]] for p in D[n][c]['pos']])
# --- RSA: cross-validated (odd/even) correlation distance
def rsa(D, c, M, perm=None, S=None):
    rs = []
    for n in D:
        if c not in D[n]: continue
        X = z(D[n][c]['X']); y = lab(D, S, n, c, perm); pos = D[n][c]['pos']
        A = []; B = []
        for k in CL:
            idx = np.where(y == k)[0]; ia = idx[0::2]; ib = idx[1::2]  # alternating split by order within class
            if len(ia) == 0 or len(ib) == 0: break
            ia = np.isin(np.arange(len(y)), ia); ib = np.isin(np.arange(len(y)), ib)
            if ia.sum() == 0 or ib.sum() == 0: break
            A.append(X[ia].mean(0)); B.append(X[ib].mean(0))
        else:
            A = np.array(A); B = np.array(B); C = np.corrcoef(A, B)[:5, 5:]; Dm = 1 - (C + C.T) / 2
            rs.append(spearmanr(Dm[iu], M[iu]).correlation)
    return float(np.nanmean(rs))
# --- geometric feature prediction (leave-one-subject-out)
def reg(D, a, b, perm=None, S=None):
    names = [n for n in D if a in D[n] and b in D[n]]; pool = [n for n in D if a in D[n]]; acc = []
    for n in names:
        tr = [m for m in pool if m != n]
        Xtr = np.vstack([z(D[m][a]['X']) for m in tr]); ytr = np.vstack([Z[[CL.index(v) for v in lab(D, S, m, a, perm)]] for m in tr])
        sc = StandardScaler().fit(Xtr); r = RidgeCV(alphas=[10, 100, 1000, 1e4, 1e5]).fit(sc.transform(Xtr), ytr)
        P = r.predict(sc.transform(z(D[n][b]['X']))); d = ((P[:, None, :] - Z[None]) ** 2).sum(-1)
        acc.append(np.mean(np.array(CL)[d.argmin(1)] == D[n][b]['y']))
    return float(np.mean(acc)), len(names)
rng = np.random.default_rng(314); P1000 = [rng.permutation(np.arange(1, 16)) for _ in range(1000)]; P300 = P1000[:300]
out = []
for sname, (D, conds, trans) in SETS.items():
    for c in conds:
        S = seqs(D, c)
        for mn, M in MOD.items():
            r = rsa(D, c, M); nl = np.array([rsa(D, c, M, p, S) for p in P1000]); p = (np.sum(nl >= r) + 1) / 1001
            out.append(dict(cluster=sname, test='RSA', condition=c, model=mn, value=r, chance95=float(np.percentile(nl, 95)), p=float(p)))
            print(f'{sname:6s} RSA {c:9s} {mn:15s} r={r:+.3f} (chance 95% {np.percentile(nl,95):+.3f}) p={p:.3f}', flush=True)
    for a, b in [(c, c) for c in conds] + trans:
        S = seqs(D, a); r, n = reg(D, a, b)
        nl = np.array(Parallel(n_jobs=2)(delayed(lambda p: reg(D, a, b, p, S)[0])(p) for p in P300)); p = (np.sum(nl >= r) + 1) / 301
        out.append(dict(cluster=sname, test='feature-estimate', condition=f'{a}->{b}', model='', value=r, chance95=float(np.percentile(nl, 95)), p=float(p), n=n))
        print(f'{sname:6s} PREDICTION {a}->{b} n={n} accuracy={r:.3f} (chance 95% {np.percentile(nl,95):.3f}) p={p:.3f}', flush=True)
ps = np.array([o['p'] for o in out]); o_ = np.argsort(ps); m = len(ps); q = np.empty(m); prev = 1
for k, i in enumerate(o_[::-1]): rank = m - k; prev = min(prev, ps[i] * m / rank); q[i] = prev
for o, qq in zip(out, q): o['q_FDR'] = float(qq)
json.dump(out, open('geo_res.json', 'w'), ensure_ascii=False)
print('FDR q<0.05:', [(o['cluster'], o['test'], o['condition'], o['model'], round(o['value'], 3)) for o in out if o['q_FDR'] < 0.05])
