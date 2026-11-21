import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '21_visual_experience_and_regions'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Vision as continuous: relation between "how well one person's brain pattern recognises the other" and vision similarity across pairs of people (Mantel)
import numpy as np,json,pickle,warnings,sys,itertools,pandas as pd;warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr,rankdata
CL=['cone','cube','sphere','pyramid','cylinder'];BANDS=['theta','alpha','beta','gamma']
TF=pickle.load(open('../../results/20_time_frequency/tf.pkl','rb'))
# vision level (from the form self-reports): 0 no vision, 1 light/large object/very little, 2 measurable low vision (~1/20–1/5), 4 sighted
VIS={'P07':4,'P11':4,'P20':0,'P08':4,'P01':2,'P09':4,'P22':4,'P03':2,'P18':2,'P14':4,'P21':2,'P19':4,'P13':1,'P17':1,'P16':1,'P05':4,'P15':2,'P06':1,'P02':2,'P10':0,'P12':4,'P04':4,'P23':0}
# years with typical vision: sighted = age; later loss = age at complete loss; congenitally limited = 0
k=pd.read_excel('../../participants/participant_information.xlsx');AGE=dict(zip(k['code'],k['age']));GR=dict(zip(k['code'],k['analysis_group']))
YRS={n:(AGE[n] if GR[n]=='Sighted' else 0) for n in AGE};YRS.update({'P20':22,'P18':15,'P21':38,'P16':14,'P23':0,'P13':0})
CONGENITAL={n:int(GR[n] in ('Congenitally low vision','Congenitally blind') or n=='P23') for n in AGE}   # congenitally limited vision
def feats(cond,bset):
    R={}
    for n,d in TF.items():
        if cond not in d: continue
        P=d[cond]['P'];bs=[BANDS.index(b) for b in bset];M=np.nanmean(P,1)[:,:,bs];ok=~np.isnan(M).any((1,2))
        X=np.log(M[ok]).reshape(ok.sum(),-1);X=(X-X.mean(0))/(X.std(0)+1e-9);R[n]=(X,np.array([CL.index(q) for q in d[cond]['y'][ok]]))
    return R
def pair(R):
    ns=sorted(R);N=len(ns);S=np.full((N,N),np.nan)
    for j,b in enumerate(ns):
        c=LogisticRegression(C=0.05,max_iter=3000).fit(*R[b])
        for i,a in enumerate(ns):
            if i==j: continue
            P=c.predict_proba(R[a][0]);S[i,j]=np.mean(P[np.arange(len(R[a][1])),R[a][1]])   # probability given to the correct shape
    return ns,S
out={}
rng=np.random.default_rng(0)
for cond,bset,tag in [('B2_img',['alpha'],'B2 imagery · alpha'),('B2_img',BANDS,'B2 imagery · all bands'),('B2_touch',['alpha'],'B2 touch · alpha'),('B2_touch',BANDS,'B2 touch · all bands')]:
    R=feats(cond,bset);ns,S=pair(R);Ss=(S+S.T)/2;iu=np.triu_indices(len(ns),1)
    res={}
    for vn,V in [('residual vision',VIS),('years with typical vision',YRS),('congenitally limited vision',CONGENITAL)]:
        v=np.array([V[n] for n in ns],float);sim=-np.abs(v[:,None]-v[None])[iu]
        r=spearmanr(Ss[iu],sim).correlation;nl=[]
        for _ in range(5000):
            p=rng.permutation(len(ns));vp=v[p];nl.append(spearmanr(Ss[iu],-np.abs(vp[:,None]-vp[None])[iu]).correlation)
        res[vn]=dict(rho=round(float(r),3),p=float((1+np.sum(np.array(nl)>=r))/5001))
    # per person: slope of "recognised by people who see less or more than them"
    v=np.array([VIS[n] for n in ns],float);eg={}
    for i,a in enumerate(ns):
        m=~np.isnan(S[i]);eg[a]=float(spearmanr(S[i][m],v[m]).correlation)
    res['person_slope']={n:round(x,3) for n,x in eg.items()}
    res['slope~own vision']=round(float(spearmanr([eg[n] for n in ns],v).correlation),3)
    out[tag]=res;print(tag,len(ns),{k_:v_ for k_,v_ in res.items() if k_!='person_slope'},flush=True)
json.dump(out,open('../../results/21_visual_experience_and_regions/mantel_res.json','w'),ensure_ascii=False,indent=1)
