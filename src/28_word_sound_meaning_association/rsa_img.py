import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '28_word_sound_meaning_association'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Imagery after the word (Block 1 imagery): does the brain pattern carry the trace of sound, meaning or association?
import numpy as np,json,pickle,itertools
from scipy.stats import spearmanr
exec(open('../../src/28_word_sound_meaning_association/rsa.py').read().split("bins=[")[0])
OL=pickle.load(open('../../data/derived/06_features/feats.pkl','rb'));R=json.load(open('../../results/16_accuracy_boosting/final_res.json'))
out={}
grp=lambda n:'CLV' if n in CLV else ('LVL' if n in LVL else 'Sighted')
# A) within-person brain dissimilarity matrix (band powers, 0.5–4.5 s), mean over people
subs=[n for n in OL if 'B1_img' in OL[n]]
NEURAL_IMG=np.array([cvrdm((lambda X:(X-X.mean(0))/(X.std(0)+1e-9))(OL[n]['B1_img']['X']),np.array([CL.index(s) for s in OL[n]['B1_img']['y']])) for n in subs])
# B) confusion matrix of the cross-subject model (final system): dissimilarity = 1 − mean confusion
C=np.zeros((5,5));c=np.zeros(5)
for n,v in R['Block 1 · imagery'].items():
    Pp=np.array(v['P']);y=np.array([CL.index(q) for q in v['y']])
    for t in range(5): C[t]+=Pp[y==t].sum(0);c[t]+=(y==t).sum()
C=C/c[:,None];Dc=np.array([1-(C[i,j]+C[j,i])/2 for i,j in P])
for mn,m in M.items():
    Mf=np.zeros((5,5))
    for (i,j),v in zip(P,m): Mf[i,j]=Mf[j,i]=v
    def sA(mm,idx=None):return np.mean([spearmanr(NEURAL_IMG[k],mm).correlation for k in range(len(subs)) if idx is None or grp(subs[k])==idx])
    oA=sA(m);nA=[sA(np.array([Mf[p[i],p[j]] for i,j in P])) for p in PERM]
    oB=spearmanr(Dc,m).correlation;nB=[spearmanr(Dc,np.array([Mf[p[i],p[j]] for i,j in P])).correlation for p in PERM]
    out[mn]=dict(person_within_rho=round(float(oA),3),person_within_p=round(float(np.mean(np.array(nA)>=oA-1e-12)),3),groups={g:round(float(sA(m,g)),3) for g in ['CLV','LVL','Sighted']},
                 confusion_rho=round(float(oB),3),confusion_p=round(float(np.mean(np.array(nB)>=oB-1e-12)),3))
    print(mn,out[mn])
json.dump(out,open('rsa_img_res.json','w'),ensure_ascii=False,indent=1)
