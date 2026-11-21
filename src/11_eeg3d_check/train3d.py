"""EEG → 3D geometry signature encoder on EEG-3D.
Channels: Quik-Cap layout derived from the data ∩ our BioSemi-64 channels. Data augmentation by left-right flipping.
Validation: never-seen person (sub12) and never-seen objects (_08, _09)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, mne, json, sys, time, glob
from scipy.signal import resample_poly
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
QUIK_M = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','M2','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']
BS = mne.channels.make_standard_montage('biosemi64').ch_names
up = {c.upper(): c for c in BS}
COMMON = [c for c in QUIK_M if c in up]                     # common channels (with EEG-3D names)
IDX3D = [QUIK_M.index(c) for c in COMMON]; OURS = [up[c] for c in COMMON]
def mirror(c):
    if c.endswith('Z'): return c
    n = int(''.join(ch for ch in c if ch.isdigit())); base = ''.join(ch for ch in c if not ch.isdigit())
    return base + str(n + 1 if n % 2 else n - 1)
FLIP = [COMMON.index(mirror(c)) if mirror(c) in COMMON else i for i, c in enumerate(COMMON)]
D = np.load('desc.npz'); NAMES = list(D['names']); FD = D['F']; MU = FD.mean(0); SD = FD.std(0) + 1e-6
FZ = (FD - MU) / SD
U, S, Vt = np.linalg.svd(FZ - FZ.mean(0), full_matrices=False); K = 12; PC = Vt[:K]  # geometry space: first 12 components
def emb(fz): e = (fz - FZ.mean(0)) @ PC.T; return e / np.linalg.norm(e, axis=-1, keepdims=True)
OBJ = {n: emb(FZ[i]) for i, n in enumerate(NAMES)}
OURS_EMB = emb((D['ours'] - MU) / SD); CL = list(D['CL'])

def load(s):
    X = np.load(f'{s}_1s.npy')[:, IDX3D]; X = resample_poly(X, 2, 5, axis=-1).astype(np.float32)  # 250→100 Hz
    lab = np.load(f'{s}_lab.npy'); nm = np.array(['_'.join(l.split('_')[1:]) for l in lab])
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-6)
    return X, nm

class Enc(nn.Module):
    def __init__(s, ch, d=K):
        super().__init__()
        s.t = nn.Sequential(nn.Conv2d(1, 40, (1, 25)), nn.AvgPool2d((1, 8), (1, 4)), nn.BatchNorm2d(40), nn.ELU(),
                            nn.Conv2d(40, 40, (ch, 1)), nn.BatchNorm2d(40), nn.ELU(), nn.Dropout(0.5), nn.Flatten())
        n = s.t(torch.zeros(1, 1, ch, 100)).shape[1]
        s.h = nn.Sequential(nn.Linear(n, 128), nn.GELU(), nn.Dropout(0.5), nn.Linear(128, d))
    def forward(s, x): z = s.h(s.t(x[:, None])); return z / z.norm(dim=-1, keepdim=True)

if __name__ == '__main__':
    subs = sorted(glob.glob('sub*_1s.npy')); subs = [s.split('_')[0] for s in subs]
    val = 'sub12' if 'sub12' in subs else subs[-1]; tr_s = [s for s in subs if s != val]
    print('common channels', len(COMMON), 'training people', tr_s, 'validation', val, flush=True)
    Xs = []; Ys = []
    for s in tr_s:
        X, nm = load(s); keep = np.array([n[-2:] not in ('08', '09') and n in OBJ for n in nm])
        Xs.append(X[keep]); Ys.append(np.stack([OBJ[n] for n in nm[keep]]))
    X = torch.tensor(np.concatenate(Xs)); Y = torch.tensor(np.concatenate(Ys)).float()
    Xv, nv = load(val); kv = np.array([n[-2:] in ('08', '09') and n in OBJ for n in nv]); Xv = torch.tensor(Xv[kv]); nv = nv[kv]
    test_objs = sorted(set(nv)); TE = torch.tensor(np.stack([OBJ[o] for o in test_objs])).float()
    yv = torch.tensor([test_objs.index(o) for o in nv])
    m = Enc(len(COMMON)); ls = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
    opt = torch.optim.AdamW(list(m.parameters()) + [ls], lr=1e-3, weight_decay=1e-2)
    EP = int(sys.argv[1]) if len(sys.argv) > 1 else 30; B = 256; best = -1; flip = torch.tensor(FLIP)
    for ep in range(EP):
        m.train(); perm = torch.randperm(len(X)); L = 0; t0 = time.time()
        for i in range(0, len(X), B):
            idx = perm[i:i + B]; x = X[idx]
            fm = torch.rand(len(idx)) < 0.5; x[fm] = x[fm][:, flip]
            z = m(x); y = Y[idx]
            loss = (1 - (z * y).sum(1)).mean()  # cosine regression
            opt.zero_grad(); loss.backward(); opt.step(); L += loss.item()
        m.eval()
        with torch.no_grad(): s = m(Xv) @ TE.T
        top1 = (s.argmax(1) == yv).float().mean().item(); top5 = (s.topk(5, 1).indices == yv[:, None]).any(1).float().mean().item()
        print(f'ep {ep} loss {L:.1f} | {val} unseen person, {len(test_objs)} unseen objects: top1 {top1:.3f} top5 {top5:.3f} (chance {1/len(test_objs):.3f}/{5/len(test_objs):.3f}) | {time.time()-t0:.0f}s', flush=True)
        if top5 > best: best = top5; torch.save(m.state_dict(), 'enc3d.pt')
    json.dump(dict(common=COMMON, ours=OURS, best_top5=best), open('enc3d_meta.json', 'w'), ensure_ascii=False)
