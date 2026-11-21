import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '06_shape_decoding_main'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, json, warnings, mne
warnings.filterwarnings('ignore'); mne.set_log_level('ERROR')
exec(open('../../src/06_shape_decoding_main/decode.py').read().split('TESTS =')[0])
from joblib import Parallel, delayed
ch=mne.read_epochs('../../data/clean_epochs/epo3/P23_block1_word_aligned-epo.fif',preload=False).ch_names
front=[i for i,c in enumerate(ch) if c.startswith(('Fp','AF','F')) and not c.startswith('FC') or c in ('FT7','FT8')]
def sel(keep):
    import copy
    for n in D:
        for c in ['B3_img','B3_erp']:
            if c in D[n]:
                D[n][c+'_full']=D[n][c].get(c+'_full',None) or dict(D[n][c])
    return
orig={n:{c:dict(D[n][c]) for c in D[n]} for n in D}
def setX(c,f):
    for n in D:
        if c in D[n]: D[n][c]['X']=f(orig[n][c]['X'])
rng=np.random.default_rng(11);PERMS=[rng.permutation(np.arange(1,16)) for _ in range(200)]
def test(lbl,c,excl=()):
    names=[n for n in D if c in D[n] and n not in excl]
    def acc(p=None):
        a=[]
        for n in names:
            tr=[m for m in names if m!=n]
            cl=clf().fit(np.vstack([z(D[m][c]['X']) for m in tr]),np.concatenate([labels(m,c,p) for m in tr]))
            a.append(cl.score(z(D[n][c]['X']),D[n][c]['y']))
        return float(np.mean(a))
    r=acc();nl=np.array(Parallel(n_jobs=2)(delayed(acc)(p) for p in PERMS))
    p=(np.sum(nl>=r)+1)/201;print(f'{lbl:45s} n={len(names)} accuracy={r:.3f} chance 95%={np.percentile(nl,95):.3f} p={p:.3f}',flush=True);return dict(acc=r,p=p,n=len(names))
out={}
keep_nog=np.arange(192)
nofront=[b*64+i for b in range(4) for i in range(64) if i not in front]
setX('B3_img',lambda X:X[:,keep_nog]);out['img_no_gamma']=test('B3 imagery – gamma band removed','B3_img')
setX('B3_img',lambda X:X[:,nofront]);out['img_without']=test('B3 imagery – forehead/frontal channels removed','B3_img')
nf2=[b*64+i for b in range(3) for i in range(64) if i not in front]
setX('B3_img',lambda X:X[:,nf2]);out['img_no_gamma_without']=test('B3 imagery – gamma and frontal channels removed','B3_img')
setX('B3_img',lambda X:X);out['img_without_P19']=test('B3 imagery – P19 removed','B3_img',('P19',))
out['erp_without_P19']=test('B3 ERP – P19 removed','B3_erp',('P19',))
nfe=[i*15+b for i in range(64) if i not in front for b in range(15)]
setX('B3_erp',lambda X:X[:,nfe]);out['erp_without']=test('B3 ERP – forehead/frontal channels removed','B3_erp')
json.dump(out,open('res_sensitivity.json','w'),ensure_ascii=False)
