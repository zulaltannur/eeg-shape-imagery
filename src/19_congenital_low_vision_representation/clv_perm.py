import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '19_congenital_low_vision_representation'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Group-label permutation: is the "training from same group − training from sighted" difference larger in the real groups?
import numpy as np,json,itertools,warnings,time,sys,pandas as pd,os;warnings.filterwarnings('ignore')
src=open('../../src/16_accuracy_boosting/final.py').read().split("out=json.load(open('final_res.json'))")[0];os.chdir('../../results/16_accuracy_boosting');exec(src)
k=pd.read_excel('../../participants/participant_information.xlsx');G=dict(zip(k['code'],k['analysis_group']))
cn=sys.argv[1];spec={'B2_img':('B2_img','B2_img'),'B2_touch':('B2_touch','B2_touch')}[cn]
R=build(*spec);names=[n for n in R if G.get(n) in ('Congenitally low vision','Sighted')]
yy={n:np.array([CL.index(q) for q in R[n]['y']]) for n in names}
cache={}
def P(tr,te):
    key=(tuple(sorted(tr)),te)
    if key not in cache: cache[key]=hier(R,R,list(tr),te,'main')
    return cache[key]
def stat(D,S,rng,ns=6):
    d=[]
    for te in D:
        cd=[m for m in D if m!=te];sub_d=[tuple(rng.choice(cd,4,replace=False)) for _ in range(ns)];sub_s=[tuple(rng.choice(S,4,replace=False)) for _ in range(ns)]
        a1=np.mean(np.mean([P(s,te) for s in sub_d],0).argmax(1)==yy[te]);a2=np.mean(np.mean([P(s,te) for s in sub_s],0).argmax(1)==yy[te]);d.append(a1-a2)
    return float(np.mean(d))
rng=np.random.default_rng(5);D=[n for n in names if G[n]=='Congenitally low vision'];S=[n for n in names if G[n]=='Sighted']
f=f'../../results/19_congenital_low_vision_representation/perm_{cn}.json';st=json.load(open(f)) if os.path.exists(f) else {'obs':stat(D,S,rng),'null':[]}
T=time.time()
while time.time()-T<540 and len(st['null'])<300:
    sh=list(rng.permutation(names));st['null'].append(stat(sh[:len(D)],sh[len(D):],rng))
    if len(st['null'])%10==0: json.dump(st,open(f,'w'))
json.dump(st,open(f,'w'));nl=np.array(st['null'])
print(cn,'observed difference',round(st['obs'],3),'n_perm',len(nl),'p',round((1+np.sum(nl>=st['obs']))/(1+len(nl)),3))
