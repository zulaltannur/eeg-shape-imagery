import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '23_three_groups_cases_connectivity'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Three groups: training source × test group (alpha, B1 and B2 imagery); training set of equal size (3 people)
import numpy as np,json,pickle,warnings,os,sys,itertools;warnings.filterwarnings('ignore')
src=open('../../src/16_accuracy_boosting/final.py').read().split("out=json.load(open('final_res.json'))")[0];os.chdir('../../results/16_accuracy_boosting');exec(src)
CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
import pandas as pd;k=pd.read_excel('../../participants/participant_information.xlsx');G0=dict(zip(k['code'],k['analysis_group']))
def G(n):return 'CLV' if n in CLV else ('LVL' if n in LVL else ('Sighted' if G0.get(n)=='Sighted' else None))
OL=pickle.load(open('../../data/derived/06_features/feats.pkl','rb'));TF=pickle.load(open('../../results/20_time_frequency/tf.pkl','rb'))
def feats(cond):
    R={}
    if cond=='B1_img':
        for n,d in OL.items():
            if cond in d and G(n): X=d[cond]['X'][:,64:128];R[n]=dict(X=(X-X.mean(0))/(X.std(0)+1e-9),y=d[cond]['y'],pos=np.array(d[cond]['pos']))
    else:
        for n,d in TF.items():
            if cond in d and G(n):
                M=np.nanmean(d[cond]['P'],1)[:,:,1];ok=~np.isnan(M).any(1);X=np.log(M[ok]);R[n]=dict(X=(X-X.mean(0))/(X.std(0)+1e-9),y=d[cond]['y'][ok],pos=d[cond]['pos'][ok])
    return R
rng=np.random.default_rng(3);perms=[rng.permutation(15) for _ in range(2000)];out={}
for cond in ['B2_img','B1_img','B2_touch']:
    R=feats(cond);yy={n:np.array([CL.index(q) for q in R[n]['y']]) for n in R};grp={g:sorted(n for n in R if G(n)==g) for g in ['CLV','LVL','Sighted']}
    print(cond,{g:len(v) for g,v in grp.items()},flush=True);out[cond]={}
    for tg in grp:
        for sg in grp:
            preds={}
            for te in grp[tg]:
                pool=[m for m in grp[sg] if m!=te];m_=min(3,len(pool))
                subs=list(itertools.combinations(pool,m_));subs=[subs[i] for i in rng.permutation(len(subs))[:15]]
                preds[te]=np.mean([hier(R,R,list(s),te,'X') for s in subs],0).argmax(1)
            obs=np.mean([np.mean(preds[n]==yy[n]) for n in preds]);nl=[]
            for pi in perms:
                a=[]
                for n in preds:
                    lab=np.full(15,-1);lab[R[n]['pos']-1]=yy[n];q=lab[pi];q=q[q>=0];a.append(np.mean(preds[n]==(q if len(q)==len(yy[n]) else rng.permutation(yy[n]))))
                nl.append(np.mean(a))
            p=float((1+np.sum(np.array(nl)>=obs))/2001)
            out[cond][f'{tg} ← {sg}']=dict(accuracy=round(float(obs),3),p=p,training_person=m_,person={n:round(float(np.mean(preds[n]==yy[n])),3) for n in preds})
            print(cond,tg,'←',sg,round(obs,3),round(p,3),flush=True)
json.dump(out,open('../../results/23_three_groups_cases_connectivity/matrix_res.json','w'),ensure_ascii=False,indent=1)
