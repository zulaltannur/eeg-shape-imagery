import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '08_coordinate_geometry'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, json
rng=np.random.default_rng(0); N=4096; R=30.0; H=60.0
def area_sample(tris,n):  # area-weighted sampling from the triangle face list
    T=np.array(tris);a=0.5*np.linalg.norm(np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]),axis=1)
    i=rng.choice(len(T),n,p=a/a.sum());u=rng.random((n,1));v=rng.random((n,1));m=u+v>1;u[m]=1-u[m];v[m]=1-v[m]
    return T[i,0]+u*(T[i,1]-T[i,0])+v*(T[i,2]-T[i,0])
def cube():
    s=R;V=np.array([(x,y,z) for x in(-s,s) for y in(-s,s) for z in(-s,s)])
    F=[(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)];t=[]
    for f in F: t+= [(V[f[0]],V[f[1]],V[f[2]]),(V[f[0]],V[f[2]],V[f[3]])]
    return area_sample(t,N)
def pyramid():
    V=np.array([(-R,-R,-R),(R,-R,-R),(R,R,-R),(-R,R,-R),(0,0,R)]);t=[(V[0],V[1],V[2]),(V[0],V[2],V[3])]+[(V[i],V[(i+1)%4],V[4]) for i in range(4)]
    return area_sample(t,N)
def sphere(): p=rng.normal(size=(N,3));return R*p/np.linalg.norm(p,axis=1,keepdims=True)
def cylinder():
    side=2*np.pi*R*H;cap=np.pi*R**2;n_s=int(N*side/(side+2*cap));th=rng.random(n_s)*2*np.pi
    S=np.c_[R*np.cos(th),R*np.sin(th),rng.uniform(-R,R,n_s)]
    k=N-n_s;r=R*np.sqrt(rng.random(k));t=rng.random(k)*2*np.pi;C=np.c_[r*np.cos(t),r*np.sin(t),np.where(rng.random(k)<.5,-R,R)]
    return np.vstack([S,C])
def cone():
    l=np.hypot(R,H);side=np.pi*R*l;cap=np.pi*R**2;n_s=int(N*side/(side+cap))
    h=H*(1-np.sqrt(rng.random(n_s)));rad=R*(1-h/H);t=rng.random(n_s)*2*np.pi;S=np.c_[rad*np.cos(t),rad*np.sin(t),h-R]
    k=N-n_s;r=R*np.sqrt(rng.random(k));t=rng.random(k)*2*np.pi;return np.vstack([S,np.c_[r*np.cos(t),r*np.sin(t),np.full(k,-R)]])
CL=['cone','cube','sphere','pyramid','cylinder'];P={'cone':cone(),'cube':cube(),'sphere':sphere(),'pyramid':pyramid(),'cylinder':cylinder()}
def chamfer(A,B):
    d=((A[:,None,:]-B[None,:,:])**2).sum(-1);return float(np.sqrt(d.min(1)).mean()+np.sqrt(d.min(0)).mean())/2
CH=np.zeros((5,5))
for i in range(5):
    for j in range(5): CH[i,j]=0 if i==j else chamfer(P[CL[i]][::4],P[CL[j]][::4])
l=np.hypot(R,H)
DESC={ # corners, edges, flat faces, curved surface, pointed tip, volume (cm3), surface area (cm2)
 'cone':[1,1,1,1,1,np.pi*R**2*H/3/1000,(np.pi*R**2+np.pi*R*l)/100],
 'cube':[8,12,6,0,0,(2*R)**3/1000,6*(2*R)**2/100],
 'sphere':[0,0,0,1,0,4/3*np.pi*R**3/1000,4*np.pi*R**2/100],
 'pyramid':[5,8,5,0,1,(2*R)**2*H/3/1000,((2*R)**2+4*0.5*2*R*l)/100],
 'cylinder':[0,2,2,1,0,np.pi*R**2*H/1000,(2*np.pi*R**2+2*np.pi*R*H)/100]}
names=['corner','edge','flat_face','curved_surface','pointed_tip','volume_cm3','surface_cm2']
M=np.array([DESC[c] for c in CL],float);Z=(M-M.mean(0))/M.std(0)
DE=np.sqrt(((Z[:,None]-Z[None])**2).sum(-1))
np.savez('geom.npz',CL=CL,chamfer=CH,desc_raw=M,desc_z=Z,desc_dist=DE,names=names,**{f'pc_{c}':P[c] for c in CL})
np.set_printoptions(precision=2,suppress=True);print('Chamfer (mm)\n',CH);print('Feature distance\n',DE);print(dict(zip(CL,M.round(1).tolist())))
iu=np.triu_indices(5,1);from scipy.stats import spearmanr;print('similarity between the two geometry models r=',round(spearmanr(CH[iu],DE[iu]).correlation,2))
