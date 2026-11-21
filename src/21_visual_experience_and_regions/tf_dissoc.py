import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '21_visual_experience_and_regions'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Time and frequency analysis: training from own group − training from sighted, difference, for congenitally low vision
import numpy as np,json,pickle,warnings,time,sys,os,pandas as pd;warnings.filterwarnings('ignore')
src=open('../../src/16_accuracy_boosting/final.py').read().split("out=json.load(open('final_res.json'))")[0];os.chdir('../../results/16_accuracy_boosting');exec(src)
k=pd.read_excel('../../participants/participant_information.xlsx');G=dict(zip(k['code'],k['analysis_group']))
OL=pickle.load(open('../../data/derived/06_features/feats.pkl','rb'));BANDS=['theta','alpha','beta','gamma']
def feats(cond,setname):
    R={}
    for n,d in OL.items():
        if G.get(n) not in ('Congenitally low vision','Sighted') or cond not in d: continue
        X=d[cond]['X'];X=X if setname=='all' else X[:,BANDS.index(setname)*64:(BANDS.index(setname)+1)*64]
        X=(X-X.mean(0))/(X.std(0)+1e-9);R[n]=dict(X=X,y=d[cond]['y'],pos=np.array(d[cond]['pos']))
    return R
cond,setname=sys.argv[1],sys.argv[2];R=feats(cond,setname);names=list(R)
yy={n:np.array([CL.index(q) for q in R[n]['y']]) for n in names}
cache={}
def P(tr,te):
    key=(tuple(sorted(tr)),te)
    if key not in cache: cache[key]=hier(R,R,list(tr),te,'X')
    return cache[key]
def run(D,S,rng,ns=6):
    a1=[];a2=[];pr1={};pr2={}
    for te in D:
        cd=[m for m in D if m!=te]
        p1=np.mean([P(tuple(rng.choice(cd,4,replace=False)),te) for _ in range(ns)],0);p2=np.mean([P(tuple(rng.choice(S,4,replace=False)),te) for _ in range(ns)],0)
        a1.append(np.mean(p1.argmax(1)==yy[te]));a2.append(np.mean(p2.argmax(1)==yy[te]));pr1[te]=p1.argmax(1);pr2[te]=p2.argmax(1)
    return float(np.mean(a1)),float(np.mean(a2)),pr1,pr2
rng=np.random.default_rng(7);D=[n for n in names if G[n]=='Congenitally low vision'];S=[n for n in names if G[n]=='Sighted']
f=f'../../results/21_visual_experience_and_regions/res_{cond}_{setname}.json'
if os.path.exists(f): st=json.load(open(f))
else:
    a1,a2,pr1,pr2=run(D,S,rng,ns=10)
    def posp(pr):   # same-order position permutation
        r2=np.random.default_rng(3);obs=np.mean([np.mean(pr[n]==yy[n]) for n in pr]);nl=[]
        for _ in range(2000):
            pi=r2.permutation(15);a=[]
            for n in pr:
                lab=np.full(15,-1);lab[R[n]['pos']-1]=yy[n];q=lab[pi];q=q[q>=0]
                a.append(np.mean(pr[n]==q) if len(q)==len(yy[n]) else np.mean(pr[n]==r2.permutation(yy[n])))
            nl.append(np.mean(a))
        return float((1+np.sum(np.array(nl)>=obs))/2001)
    st=dict(same_group=a1,same_group_p=posp(pr1),sighted=a2,sighted_p=posp(pr2),diff=a1-a2,null=[],n_D=len(D),n_S=len(S))
T=time.time()
while time.time()-T<float(sys.argv[3]) and len(st['null'])<200:
    sh=list(rng.permutation(D+S));b1,b2,_,_=run(sh[:len(D)],sh[len(D):],rng);st['null'].append(b1-b2)
    if len(st['null'])%10==0: json.dump(st,open(f,'w'))
json.dump(st,open(f,'w'));nl=np.array(st['null'])
print(cond,setname,'same',round(st['same_group'],3),round(st['same_group_p'],3),'sighted',round(st['sighted'],3),'diff',round(st['diff'],3),'n',len(nl),'p',round((1+np.sum(nl>=st['diff']))/(1+len(nl)),3),flush=True)
