import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '10_multi_component_model'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,json,warnings;warnings.filterwarnings('ignore')
exec(open('../../src/10_multi_component_model/fit.py').read().split('rng=np.random')[0])
c='B1_erp';T0=np.arange(-0.1,0.75,0.05)+0.05;nw=D['P15'][c]['T'].shape[2];S=seqs(c)
def bt(w,perm=None):
    out=[]
    for n in D:
        if c not in D[n]: continue
        y=D[n][c]['y'] if perm is None else np.array([S[n][perm[p-1]] for p in D[n][c]['pos']])
        v=nrdm(z(D[n][c]['T'][:,:,w]),y)
        if v is not None: out.append(np.linalg.lstsq(np.c_[np.ones(10),M],v,rcond=None)[0][1:])
    return np.mean(out,0)
real=np.array([bt(w) for w in range(nw)])
rng=np.random.default_rng(9);P=[rng.permutation(np.arange(1,16)) for _ in range(500)]
null=np.array([[bt(w,p) for w in range(nw)] for p in P])  # perm,w,comp
mx=null[:,:,2].max(1)
for w in range(nw):
    pu=(np.sum(null[:,w,2]>=real[w,2])+1)/501;pf=(np.sum(mx>=real[w,2])+1)/501
    print(f'{T0[w]*1000:5.0f} ms  geometry β={real[w,2]:+.3f} p={pu:.3f} (p corrected across windows={pf:.3f})  word β={real[w,3]:+.3f}')
json.dump(dict(t=T0[:nw].tolist(),beta=real.tolist()),open('timefit_res.json','w'))
