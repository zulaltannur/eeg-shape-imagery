import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '11_eeg3d_check'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, glob, os
rng=np.random.default_rng(0)
def sig(P):
    P=P[:,:3]-P[:,:3].mean(0); P=P/np.linalg.norm(P,axis=1).max()
    i=rng.integers(len(P),size=(30000,3))
    d2=np.linalg.norm(P[i[:,0]]-P[i[:,1]],axis=1)
    a=P[i[:,0]]-P[i[:,1]];b=P[i[:,2]]-P[i[:,1]];ang=np.arccos(np.clip((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-9),-1,1))
    rad=np.linalg.norm(P,axis=1)
    ev=np.sort(np.linalg.eigvalsh(np.cov(P.T)))[::-1];ev=ev/ev.sum()
    h=lambda x,lo,hi,n: np.histogram(x,n,(lo,hi))[0]/len(x)
    return np.concatenate([h(d2,0,2,24),h(ang,0,np.pi,12),h(rad,0,1,12),ev,[ev[1]/ev[0],ev[2]/ev[0]]])
F={os.path.basename(f)[:-4]:sig(np.load(f)) for f in sorted(glob.glob('point_cloud_simple/*.npy'))}
g=np.load('../../results/08_coordinate_geometry/geom.npz');CL=list(g['CL']);O={c:sig(g[f'pc_{c}']) for c in CL}
np.savez('desc.npz',names=list(F),F=np.array(list(F.values())),ours=np.array([O[c] for c in CL]),CL=CL)
A=np.array(list(F.values()));mu=A.mean(0);sd=A.std(0)+1e-6;Z=(A-mu)/sd;Oz=(np.array([O[c] for c in CL])-mu)/sd
nm=list(F)
for k,c in enumerate(CL):
    d=np.linalg.norm(Z-Oz[k],axis=1);o=np.argsort(d)[:5];print(c,'→ nearest EEG-3D objects:',[nm[i] for i in o])
D=np.linalg.norm(Oz[:,None]-Oz[None],axis=-1);np.set_printoptions(precision=1,suppress=True);print(CL);print(D)
