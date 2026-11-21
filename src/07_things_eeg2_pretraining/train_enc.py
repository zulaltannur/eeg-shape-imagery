"""EEG→CLIP encoder on THINGS-EEG2 (NICE-style spatio-temporal convolution, contrastive learning).
Cross-subject: sub-01..09 for training, sub-10 for validation (never-seen person, 200 test images)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, glob, sys, time, json
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2)
SUBS = sorted(glob.glob('sub-*_train.npy')); VAL = 'sub-10'
Etr = torch.tensor(np.load('clip_training_images.npy')); Ete = torch.tensor(np.load('clip_test_images.npy'))

def norm(X):  # within-person channel z-score (the same is applied to our data)
    m = X.mean(axis=(0, 1, 3), keepdims=True) if X.ndim == 4 else X.mean(axis=(0, 2), keepdims=True)
    s = X.std(axis=(0, 1, 3), keepdims=True) if X.ndim == 4 else X.std(axis=(0, 2), keepdims=True)
    return ((X - m) / (s + 1e-6)).astype(np.float32)

class Enc(nn.Module):
    def __init__(s, ch=17, d=512):
        super().__init__()
        s.t = nn.Sequential(nn.Conv2d(1, 40, (1, 25)), nn.AvgPool2d((1, 8), (1, 4)), nn.BatchNorm2d(40), nn.ELU(),
                            nn.Conv2d(40, 40, (ch, 1)), nn.BatchNorm2d(40), nn.ELU(), nn.Dropout(0.5))
        s.p = nn.Sequential(nn.Conv2d(40, 40, 1), nn.Flatten())
        n = s.p(s.t(torch.zeros(1, 1, ch, 100))).shape[1]
        s.h = nn.Sequential(nn.Linear(n, d), nn.GELU(), nn.Dropout(0.5), nn.Linear(d, d), nn.LayerNorm(d))
    def forward(s, x): z = s.h(s.p(s.t(x[:, None]))); return z / z.norm(dim=-1, keepdim=True)

if __name__ == '__main__':
    tr = [f for f in SUBS if VAL not in f]
    N = 16540; K = 3  # repetition mean + 2 single repetitions
    X = np.empty((len(tr) * K * N, 17, 100), np.float16); Y = np.empty(len(tr) * K * N, np.int64); k = 0
    for f in tr:
        x = np.load(f, mmap_mode='r')
        x = np.asarray(x, dtype=np.float32)
        m_ = x.mean(axis=(0, 1, 3), keepdims=True); s_ = x.std(axis=(0, 1, 3), keepdims=True); x = (x - m_) / (s_ + 1e-6)
        for arr in (x.mean(1), x[:, 0], x[:, 1]):
            X[k:k + N] = arr; Y[k:k + N] = np.arange(N); k += N
        del x; print('loaded', f, flush=True)
    X = torch.from_numpy(X); Y = torch.from_numpy(Y)
    xv = torch.tensor(norm(np.load(f'{VAL}_test.npy').astype(np.float32)).mean(1))  # (200.17.100)
    print('training samples', X.shape, flush=True)
    m = Enc(); logit_s = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
    opt = torch.optim.AdamW(list(m.parameters()) + [logit_s], lr=3e-4, weight_decay=1e-2)
    EP = int(sys.argv[1]) if len(sys.argv) > 1 else 12; B = 512; hist = []
    for ep in range(EP):
        m.train(); perm = torch.randperm(len(X)); t0 = time.time(); L = 0
        for i in range(0, len(X), B):
            idx = perm[i:i + B]; z = m(X[idx].float()); e = Etr[Y[idx]]
            lg = logit_s.exp() * z @ e.T; lab = torch.arange(len(idx))
            loss = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step(); L += loss.item()
        m.eval()
        with torch.no_grad(): s = m(xv) @ Ete.T
        top1 = (s.argmax(1) == torch.arange(200)).float().mean().item(); top5 = (s.topk(5, 1).indices == torch.arange(200)[:, None]).any(1).float().mean().item()
        hist.append(dict(ep=ep, loss=L, top1=top1, top5=top5)); print(f'ep {ep} loss {L:.1f} | sub-10 (unseen person) on 200 images top1 {top1:.3f} top5 {top5:.3f} (chance 0.005/0.025) | {time.time()-t0:.0f}s', flush=True)
        torch.save(m.state_dict(), 'enc.pt')
    json.dump(hist, open('enc_hist.json', 'w'))
