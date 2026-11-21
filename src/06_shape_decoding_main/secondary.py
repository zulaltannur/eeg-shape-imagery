import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '06_shape_decoding_main'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, pickle, json, warnings, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
exec(open('../../src/06_shape_decoding_main/decode.py').read().split('TESTS =')[0])  # D, SEQ, labels, clf, run, z
P = pd.read_excel('../../participants/participant_information.xlsx'); G = dict(zip(P.code, P.analysis_group))
out = {}
# 1) Discrimination over time (ERP)
def tr_run(cond, w, perm=None):
    names = [n for n in D if cond in D[n]]
    acc = []
    for n in names:
        tr = [m for m in names if m != n]
        Xtr = np.vstack([z(D[m][cond]['T'][:, :, w]) for m in tr]); ytr = np.concatenate([labels(m, cond, perm) for m in tr])
        acc.append(clf().fit(Xtr, ytr).score(z(D[n][cond]['T'][:, :, w]), D[n][cond]['y']))
    return float(np.mean(acc))
rng = np.random.default_rng(7); PERMS = [rng.permutation(np.arange(1, 16)) for _ in range(200)]
times = np.arange(-0.1, 0.75, 0.05) + 0.05  # window centre
for cond in ['B3_erp', 'B1_erp']:
    nw = D['P08' if cond == 'B3_erp' else 'P15'][cond]['T'].shape[2]
    real = [tr_run(cond, w) for w in range(nw)]
    null = np.array(Parallel(n_jobs=2)(delayed(lambda p: [tr_run(cond, w, p) for w in range(nw)])(p) for p in PERMS))
    mx = null.max(1)  # maximum over windows -> multiple comparison correction
    p_fwe = [(np.sum(mx >= r) + 1) / (len(PERMS) + 1) for r in real]
    p_unc = [(np.sum(null[:, i] >= r) + 1) / (len(PERMS) + 1) for i, r in enumerate(real)]
    out[f'time_{cond}'] = dict(t=times[:nw].tolist(), acc=real, null_mean=null.mean(0).tolist(), null95=np.percentile(null, 95, 0).tolist(), p_fwe=p_fwe, p_unc=p_unc)
    print(cond, [f'{t:.2f}:{a:.2f}{"*" if pf < .05 else ("+" if pu < .05 else "")}' for t, a, pf, pu in zip(times, real, p_fwe, p_unc)], flush=True)
# 2) Confusion matrices and per-person accuracy (true labels)
for a, b in [('B3_erp', 'B3_erp'), ('B3_img', 'B3_img'), ('B3_perc', 'B3_img'), ('B2_touch', 'B2_touch'), ('B1_img', 'B1_img')]:
    names = [n for n in D if a in D[n] and b in D[n]]; Y = []; Pr = []
    for n in names:
        tr = [m for m in D if a in D[m] and m != n]
        c = clf().fit(np.vstack([z(D[m][a]['X']) for m in tr]), np.concatenate([D[m][a]['y'] for m in tr]))
        Y += list(D[n][b]['y']); Pr += list(c.predict(z(D[n][b]['X'])))
    out[f'confusion_{a}->{b}'] = confusion_matrix(Y, Pr, labels=CL).tolist()
# 3) Weight patterns (Haufe transform), model trained on all people
for cond in ['B3_erp', 'B3_img', 'B2_touch', 'B1_img']:
    names = [n for n in D if cond in D[n]]
    X = np.vstack([z(D[n][cond]['X']) for n in names]); y = np.concatenate([D[n][cond]['y'] for n in names])
    c = clf().fit(X, y); W = c[-1].coef_ / c[0].scale_
    Xc = X - X.mean(0); A = (Xc.T @ Xc / len(X)) @ W.T  # feature × class
    out[f'pattern_{cond}'] = np.abs(A).mean(1).tolist()
# 4) Within person (leave one trial out), for information only
wi = {}
for cond in ['B1_img', 'B2_touch', 'B2_img', 'B3_erp', 'B3_perc', 'B3_img']:
    accs = {}
    for n in D:
        if cond not in D[n]: continue
        X = z(D[n][cond]['X']); y = D[n][cond]['y']; hit = []
        for i in range(len(y)):
            m = np.arange(len(y)) != i
            hit.append(clf().fit(X[m], y[m]).predict(X[i:i+1])[0] == y[i])
        accs[n] = float(np.mean(hit))
    wi[cond] = accs; print('within person', cond, round(np.mean(list(accs.values())), 3), flush=True)
out['person_within'] = wi
# 5) Transfer between groups: train on sighted -> test on visually impaired, and the reverse
gx = {}
for cond in ['B1_img', 'B2_touch', 'B2_img']:
    S = [n for n in D if cond in D[n] and G[n] == 'Sighted']; V = [n for n in D if cond in D[n] and G[n] != 'Sighted']
    for nm, tr, te in [('sighted→impaired', S, V), ('impaired→sighted', V, S)]:
        c = clf().fit(np.vstack([z(D[m][cond]['X']) for m in tr]), np.concatenate([D[m][cond]['y'] for m in tr]))
        gx[f'{cond} {nm}'] = float(np.mean([c.score(z(D[n][cond]['X']), D[n][cond]['y']) for n in te]))
out['group_transfer'] = gx; print(gx)
json.dump(out, open('../../results/06_shape_decoding_main/res_secondary.json', 'w'), ensure_ascii=False)
