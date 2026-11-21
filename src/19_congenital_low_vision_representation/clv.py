import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '19_congenital_low_vision_representation'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Stage 19: representation of tactile imagery in people with congenitally low vision
import numpy as np,json,pickle,itertools,warnings,pandas as pd;warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
src=open('../../src/16_accuracy_boosting/final.py').read().split("out=json.load(open('final_res.json'))")[0]
import os;os.chdir('../../results/16_accuracy_boosting');exec(src)       # build(), hier(), CL, BL
rng=np.random.default_rng(1)
k=pd.read_excel('../../participants/participant_information.xlsx');G=dict(zip(k['code'],k['analysis_group']))
def grp(names,g):return [n for n in names if G.get(n)==g]
COND={'B2 imagery':('B2_img','B2_img'),'B2 touch to imagery':('B2_touch','B2_touch','B2_img','B2_img'),'B2 touch':('B2_touch','B2_touch'),'B1 imagery':('B1_img','B1_img')}
perms=[rng.permutation(15) for _ in range(2000)]
def permacc(preds):   # preds: {n:(pred,y,pos)} -> observed, p (same-order position permutation)
    obs=np.mean([np.mean(p==y) for p,y,_ in preds.values()]);nl=[]
    for pi in perms:
        a=[]
        for p,y,pos in preds.values():
            lab=np.full(15,-1);lab[pos-1]=y;q=lab[pi];q=q[q>=0]
            if len(q)!=len(y): q=y[rng.permutation(len(y))]
            a.append(np.mean(p==q))
        nl.append(np.mean(a))
    nl=np.array(nl);return float(obs),float((1+np.sum(nl>=obs))/(1+len(nl))),float(np.percentile(nl,95))
out={}
# ---------- A. training source × test group ----------
A={}
for cn,spec in COND.items():
    R=build(*spec[:2]);RT=build(*spec[2:]) if len(spec)>2 else R;names=[n for n in R if n in RT]
    D=grp(names,'Congenitally low vision');S=grp(names,'Sighted');L=grp(names,'Later vision loss')
    A[cn]={}
    for f in ['main','combined']:
        cells={}
        for tname,test,src_name,pool in [('Congenitally low vision','D','Congenitally low vision',D),('Congenitally low vision','D','Sighted',S),('Sighted','S','Sighted',S),('Sighted','S','Congenitally low vision',D),
                                         ('Congenitally low vision','D','Later vision loss',L),('Sighted','S','Later vision loss',L)]:
            tests=D if test=='D' else S;preds={};pk={}
            for te in tests:
                cand=[m for m in pool if m!=te];m_=min(len(D)-1,len(cand),4 if 'Later' in src_name else 5)
                subs=list(itertools.combinations(cand,m_));subs=[subs[i] for i in rng.permutation(len(subs))[:20]]
                P=np.mean([hier(R,RT,list(s),te,f) for s in subs],0)
                yy=np.array([CL.index(q) for q in RT[te]['y']]);preds[te]=(P.argmax(1),yy,np.array(RT[te]['pos']));pk[te]=float(np.mean(preds[te][0]==preds[te][1]))
            o,p,n95=permacc(preds)
            cells[f'{tname} ← {src_name}']=dict(accuracy=round(o,3),p=p,null95=round(n95,3),training_person=m_,person=pk)
            print(cn,f,tname,'←',src_name,round(o,3),round(p,3),flush=True)
        A[cn][f]=cells
out['A_training_source']=A
# ---------- B. representational geometry (within person, cross-validated) ----------
gd=np.load('../../results/08_coordinate_geometry/geom.npz',allow_pickle=True);gc=list(gd['CL']);ix=[gc.index(c) for c in CL]
desc=gd['desc_raw'][ix];V=np.load('../../data/derived/07_things/clip_shape.npy')
def zc(X):return (X-X.mean(0))/(X.std(0)+1e-9)
iu=np.triu_indices(5,1)
MOD={'visual (CLIP)':(1-np.corrcoef(V))[iu],
     'geometric (Chamfer)':gd['chamfer'][np.ix_(ix,ix)][iu],
     'tactile structure (corner, edge, flat face, curved surface, pointed tip)':np.array([np.linalg.norm(a-b) for a,b in itertools.combinations(zc(desc[:,:5]),2)]),
     'size (volume)':np.array([abs(a-b) for a,b in itertools.combinations(desc[:,5],2)]),
     'curved/cornered':np.array([float(a!=b) for a,b in itertools.combinations([1,0,1,0,1],2)]),
     'pointed/not':np.array([float(a!=b) for a,b in itertools.combinations([1,0,0,1,0],2)])}
def rdm(X,y):
    X=(X-X.mean(0))/(X.std(0)+1e-9);occ=np.zeros(len(y),int)
    for t in range(5): occ[y==t]=np.arange((y==t).sum())
    Ds=[]
    for kk in range(occ.max()+1):
        A_=[X[(y==t)&(occ==kk)].mean(0) if ((y==t)&(occ==kk)).any() else None for t in range(5)]
        B_=[X[(y==t)&(occ!=kk)].mean(0) if ((y==t)&(occ!=kk)).any() else None for t in range(5)]
        D=np.full((5,5),np.nan)
        for i in range(5):
            for j in range(5):
                if A_[i] is not None and B_[j] is not None: D[i,j]=1-np.corrcoef(A_[i],B_[j])[0,1]
        Ds.append(D)
    D=np.nanmean(Ds,0);return (D+D.T)/2
B={}
P120=list(itertools.permutations(range(5)))
def permmod(m,p):
    M=np.zeros((5,5));M[iu]=m;M=M+M.T;return M[np.ix_(p,p)][iu]
for cn,(a,b) in {'B2 imagery':('B2_img','B2_img'),'B2 touch':('B2_touch','B2_touch'),'B1 imagery':('B1_img','B1_img')}.items():
    R=build(a,b);B[cn]={}
    for g in ['Congenitally low vision','Sighted','Later vision loss']:
        ns=grp(list(R),g);Dn={n:rdm(R[n]['main'],np.array([CL.index(q) for q in R[n]['y']])) for n in ns}
        discrimination={n:float(np.nanmean(D[iu])-np.nanmean(np.diag(D))) for n,D in Dn.items()}
        res={'n':len(ns),'discriminability':round(float(np.mean(list(discrimination.values()))),4),'person_discrimination':{n:round(v,4) for n,v in discrimination.items()},'model':{}}
        for mn,m in MOD.items():
            r=np.mean([spearmanr(D[iu],m).correlation for D in Dn.values()])
            nl=[np.mean([spearmanr(D[iu],permmod(m,list(p))).correlation for D in Dn.values()]) for p in P120]
            res['model'][mn]=dict(rho=round(float(r),3),p=float(np.mean(np.array(nl)>=r-1e-12)))
        B[cn][g]=res;print(cn,g,res['n'],res['discriminability'],{k_:v['rho'] for k_,v in res['model'].items()},flush=True)
out['B_representation_geometry']=B
json.dump(out,open('../../results/19_congenital_low_vision_representation/clv_res.json','w'),ensure_ascii=False,indent=1)
