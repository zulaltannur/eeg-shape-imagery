"""Apply the THINGS-pretrained encoder to all our blocks (exploratory analysis)."""
import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '07_things_eeg2_pretraining'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, torch, pickle, json, glob, open_clip, warnings
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
warnings.filterwarnings('ignore'); torch.set_num_threads(2)
from train_enc import Enc
CL = ['cone', 'cube', 'sphere', 'pyramid', 'cylinder']
EN_NAME = {'cone': 'cone', 'cube': 'cube', 'sphere': 'sphere', 'pyramid': 'pyramid', 'cylinder': 'cylinder'}
# target representations: matte red 3D shapes drawn after the real objects (4 angles) + text
cm, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'); cm.eval(); tok = open_clip.get_tokenizer('ViT-B-32')
with torch.no_grad():
    IMG = []; TXT = []
    for c in CL:
        e = cm.encode_image(torch.stack([pre(Image.open(f).convert('RGB')) for f in sorted(glob.glob(f'shapes/{c}_*.png'))]))
        e = e / e.norm(dim=-1, keepdim=True); IMG.append(e.mean(0))
        t = cm.encode_text(tok([f'a photo of a red {EN_NAME[c]}', f'a 3D render of a {EN_NAME[c]}', f'a {EN_NAME[c]}'])); t = t / t.norm(dim=-1, keepdim=True); TXT.append(t.mean(0))
    IMG = torch.stack(IMG); IMG = IMG / IMG.norm(dim=-1, keepdim=True); TXT = torch.stack(TXT); TXT = TXT / TXT.norm(dim=-1, keepdim=True)
TGT = {'drawing': IMG.numpy(), 'text': TXT.numpy(), 'both': ((IMG + TXT) / 2).numpy()}
enc = Enc(); enc.load_state_dict(torch.load('enc.pt')); enc.eval()
D = pickle.load(open('ours.pkl', 'rb'))
E = {}
with torch.no_grad():
    for n, d in D.items():
        for c, v in d.items():
            S = v['S'].astype(np.float32)  # (trial, seg, 17, 100)
            m = S.mean(axis=(0, 1, 3), keepdims=True); s = S.std(axis=(0, 1, 3), keepdims=True); S = (S - m) / (s + 1e-6)
            z = enc(torch.tensor(S.reshape(-1, 17, 100))).reshape(S.shape[0], S.shape[1], -1).mean(1)
            z = z / z.norm(dim=-1, keepdim=True)
            E.setdefault(n, {})[c] = dict(Z=z.numpy(), y=np.array(v['y']), pos=np.array(v['pos']))
pickle.dump(E, open('ours_emb.pkl', 'wb'))
CONDS = ['B1_word', 'B1_img', 'B2_touch', 'B2_img', 'B3_picture', 'B3_viewing', 'B3_img']
def seqs(c):
    S = {n: dict(zip(E[n][c]['pos'].tolist(), E[n][c]['y'].tolist())) for n in E if c in E[n]}; com = {}
    for n, m in S.items():
        if not (n == 'P23' and c.startswith('B2')): [com.setdefault(p, l) for p, l in m.items()]
    for n in S:
        b = dict(com) if not (n == 'P23' and c.startswith('B2')) else {15: 'pyramid'}; b.update(S[n]); S[n] = b
    return S
SQ = {c: seqs(c) for c in CONDS}
def lab(n, c, perm=None): return E[n][c]['y'] if perm is None else np.array([SQ[c][n][perm[p - 1]] for p in E[n][c]['pos']])
rng = np.random.default_rng(2027); PERMS = [rng.permutation(np.arange(1, 16)) for _ in range(500)]
out = {}
# 1) direct (zero-shot): no training
def zs(c, T, perm=None):
    acc = []; rk = []
    for n in E:
        if c not in E[n]: continue
        sim = E[n][c]['Z'] @ T.T; y = lab(n, c, perm); yi = np.array([CL.index(v) for v in y])
        acc.append(np.mean(sim.argmax(1) == yi)); rk.append(np.mean([(sim[i] > sim[i, yi[i]]).sum() + 1 for i in range(len(yi))]))
    return float(np.mean(acc)), float(np.mean(rk))
for c in CONDS:
    for tn, T in TGT.items():
        a, r = zs(c, T); nl = np.array([zs(c, T, p) for p in PERMS])
        pa = (np.sum(nl[:, 0] >= a) + 1) / 501; pr = (np.sum(nl[:, 1] <= r) + 1) / 501
        out[f'direct|{c}|{tn}'] = dict(acc=a, rank=r, null_acc95=float(np.percentile(nl[:, 0], 95)), p_acc=float(pa), p_rank=float(pr))
        print(f'direct {c:9s} {tn:5s} accuracy={a:.3f} (chance 95% {np.percentile(nl[:,0],95):.3f}, p={pa:.3f}) mean.rank={r:.2f} (p={pr:.3f})', flush=True)
# 2) fine-tuned: leave-one-subject-out on the encoder output
def z_(X): return (X - X.mean(0)) / (X.std(0) + 1e-9)
def ft(a, b, perm=None):
    names = [n for n in E if a in E[n] and b in E[n]]; pool = [n for n in E if a in E[n]]; acc = []
    for n in names:
        tr = [m for m in pool if m != n]
        cl = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(np.vstack([z_(E[m][a]['Z']) for m in tr]), np.concatenate([lab(m, a, perm) for m in tr]))
        acc.append(cl.score(z_(E[n][b]['Z']), E[n][b]['y']))
    return float(np.mean(acc)), len(names)
for a, b in [(c, c) for c in CONDS] + [('B2_touch', 'B2_img'), ('B3_viewing', 'B3_img'), ('B3_picture', 'B3_img'), ('B3_picture', 'B2_touch'), ('B3_picture', 'B1_img'), ('B3_picture', 'B2_img')]:
    r, k = ft(a, b); nl = np.array(Parallel(n_jobs=2)(delayed(lambda p: ft(a, b, p)[0])(p) for p in PERMS))
    p = (np.sum(nl >= r) + 1) / 501
    out[f'finetuned|{a}->{b}'] = dict(acc=r, n=k, null95=float(np.percentile(nl, 95)), p=float(p))
    print(f'fine-tuned {a:9s}->{b:9s} n={k:2d} accuracy={r:.3f} (chance 95% {np.percentile(nl,95):.3f}) p={p:.3f}', flush=True)
json.dump(out, open('apply_res.json', 'w'), ensure_ascii=False)
