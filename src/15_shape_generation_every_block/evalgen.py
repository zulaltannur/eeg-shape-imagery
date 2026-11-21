import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '15_shape_generation_every_block'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,json,sys,warnings;warnings.filterwarnings('ignore')
sys.path.insert(0,'../../src/09_superquadric_bridge');from sq import PAR,surface
CL=['cone','cube','sphere','pyramid','cylinder'];g=np.load('../../results/08_coordinate_geometry/geom.npz')
TRUE={c:g[f'pc_{c}'][::8]/60.0 for c in CL}  # real CAD point clouds (unit: 60 mm)
def sq_points(p):
    X,Y,Z=surface(*p,n=40);P=np.c_[X.ravel(),Y.ravel(),Z.ravel()];P=P-P.mean(0)
    return P
def chamfer(A,B):
    d=((A[:,None]-B[None])**2).sum(-1);return float(np.sqrt(d.min(1)).mean()+np.sqrt(d.min(0)).mean())/2
TP={c:TRUE[c]-TRUE[c].mean(0) for c in CL}
PR=np.array([PAR[c] for c in CL])
cache={}
def gen_shape(P):  # probability-weighted superquadric
    return P@PR
r=json.load(open('gen_res.json'));out=json.load(open('gen_eval.json'));rng=np.random.default_rng(11)
seq={}
for name,res in r.items():
    for n,v in res.items():
        if n!='P23' or 'Block 2' not in name:
            for p,y in zip(v['pos'],v['y']): seq.setdefault((name,p),y)
for name,res in r.items():
    if name in out: continue
    pred={n:np.array(v['P']) for n,v in res.items()};Y={n:np.array(v['y']) for n,v in res.items()};pos={n:v['pos'] for n,v in res.items()}
    # shape of every generated trial and coordinate distance to the true shape
    D={n:np.array([[chamfer(sq_points(gen_shape(P)),TP[c]) for c in CL] for P in pred[n]]) for n in pred}  # trial x possible true shape
    def metrics(lab):
        acc=np.mean([np.mean(np.array(CL)[pred[n].argmax(1)]==lab[n]) for n in pred])
        ch=np.mean([np.mean([D[n][i,CL.index(l)] for i,l in enumerate(lab[n])]) for n in pred])
        return acc,ch
    acc,ch=metrics(Y);NA=[];NC=[]
    for _ in range(2000):
        pm=rng.permutation(np.arange(1,16))
        lab={n:np.array([ (seq.get((name,pm[p-1])) if not (n=='P23' and 'Block 2' in name) else dict(zip(pos[n],Y[n])).get(pm[p-1],'pyramid')) or Y[n][i] for i,p in enumerate(pos[n])]) for n in pred}
        a,c=metrics(lab);NA.append(a);NC.append(c)
    pa=(np.sum(np.array(NA)>=acc)+1)/2001;pc=(np.sum(np.array(NC)<=ch)+1)/2001
    out[name]=dict(acc=acc,p_acc=pa,null_acc95=float(np.percentile(NA,95)),chamfer=ch,null_ch=float(np.mean(NC)),null_ch05=float(np.percentile(NC,5)),p_chamfer=pc,
                   mean_params={c:np.mean(np.vstack([gen_shape(pred[n][Y[n]==c]) for n in pred if (Y[n]==c).any()]),0).tolist() for c in CL})
    print(f'{name:24s} accuracy {acc:.3f} (chance 95% {np.percentile(NA,95):.3f}, p={pa:.3f}) | coordinate distance {ch*60:.2f} mm (chance {np.mean(NC)*60:.2f} mm, p={pc:.3f})',flush=True)
json.dump(out,open('gen_eval.json','w'),ensure_ascii=False)
