"""Shape-constrained hidden Markov model: shared emissions, shape-specific transitions. Leave-one-subject-out shape test."""
import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '13_tactile_hmm_touchsim'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,pickle,json,warnings;warnings.filterwarnings('ignore')
from scipy.special import logsumexp
D=pickle.load(open('hmm_feats_rel.pkl','rb'));NAMES=sorted(D)
ST=['flat surface','curved surface','edge','corner/tip','all hand grasp'];K=5
MASK={'sphere':[0,1,0,0,1],'cube':[1,0,1,1,1],'cylinder':[1,1,1,0,1],'cone':[1,1,1,1,1],'pyramid':[1,0,1,1,1]}
CL=list(MASK);MASK={k:np.array(v,bool) for k,v in MASK.items()}
def loglik_em(F,ok,mu,present):
    L=-0.5*(((F[:,None,:]-mu[None])**2)/present[None]+np.log(2*np.pi*present[None])).sum(-1)
    L[~ok]=0.0;return L
def fb(L,lA,lpi,m):
    T=len(L);L=np.where(m[None],L,-np.inf);la=np.zeros((T,K));la[0]=lpi+L[0]
    for t in range(1,T): la[t]=logsumexp(la[t-1][:,None]+lA,0)+L[t]
    ll=logsumexp(la[-1]);lb=np.zeros((T,K))
    for t in range(T-2,-1,-1): lb[t]=logsumexp(lA+(L[t+1]+lb[t+1])[None],1)
    g=np.exp(la+lb-ll);xi=np.zeros((K,K))
    for t in range(T-1): xi+=np.exp(la[t][:,None]+lA+(L[t+1]+lb[t+1])[None]-ll)
    return ll,g,xi
def masked_log(A,m):
    A=A*m[None]*m[:,None]+1e-12;A=A/A.sum(1,keepdims=True);return np.log(A)
def fit(names,iters=25,seed=0):
    rng=np.random.default_rng(seed);allF=np.vstack([t['F'][t['ok']] for n in names for t in D[n]['trials']])
    mu=allF[rng.choice(len(allF),K,replace=False)];present=np.ones((K,allF.shape[1]))
    A={s:np.full((K,K),1/K)+np.eye(K)*2 for s in CL};pi={s:np.ones(K) for s in CL}
    for it in range(iters):
        S0=np.zeros(K);S1=np.zeros_like(mu);S2=np.zeros_like(mu);XI={s:np.full((K,K),0.5)+np.eye(K) for s in CL};PI={s:np.full(K,0.5) for s in CL};tot=0
        for n in names:
            for t in D[n]['trials']:
                s=t['shape'];m=MASK[s];L=loglik_em(t['F'],t['ok'],mu,present)
                ll,g,xi=fb(L,masked_log(A[s],m),np.log(pi[s]*m+1e-12)-np.log((pi[s]*m).sum()),m);tot+=ll
                go=g[t['ok']];Fo=t['F'][t['ok']];S0+=go.sum(0);S1+=go.T@Fo;S2+=go.T@(Fo**2);XI[s]+=xi;PI[s]+=g[0]
        mu=S1/S0[:,None];present=np.maximum(S2/S0[:,None]-mu**2,0.05)
        for s in CL: A[s]=XI[s]/XI[s].sum(1,keepdims=True);pi[s]=PI[s]/PI[s].sum()
    return dict(mu=mu,present=present,A=A,pi=pi,ll=tot)
def score(model,t,s):
    m=MASK[s];L=loglik_em(t['F'],t['ok'],model['mu'],model['present'])
    return fb(L,masked_log(model['A'][s],m),np.log(model['pi'][s]*m+1e-12)-np.log((model['pi'][s]*m).sum()),m)[0]
if __name__ == '__main__':
    import os
    ck='hmm_ck_rel.pkl';pred,true,pos=pickle.load(open(ck,'rb')) if os.path.exists(ck) else ({},{},{})
    for n in NAMES:
        if n in pred: continue
        tr=[x for x in NAMES if x!=n];M=fit(tr)
        P=[];Y=[];Q=[]
        for t in D[n]['trials']:
            sc=np.array([score(M,t,s) for s in CL]);P.append(CL[int(sc.argmax())]);Y.append(t['shape']);Q.append(t['trial'])
        pred[n]=P;true[n]=Y;pos[n]=Q;pickle.dump((pred,true,pos),open(ck,'wb'));print(n,'accuracy',round(np.mean(np.array(P)==np.array(Y)),2),flush=True)
    acc=np.mean([np.mean(np.array(pred[n])==np.array(true[n])) for n in NAMES])
    # chance: the same shuffle of trial positions for everyone (design structure preserved)
    seq={}
    for n in NAMES:
        for p,y in zip(pos[n],true[n]):
            if not(n=='P23'): seq.setdefault(p,y)
    rng=np.random.default_rng(3);null=[]
    for _ in range(2000):
        pm=rng.permutation(np.arange(1,16));a=[]
        for n in NAMES:
            sq={p:y for p,y in zip(pos[n],true[n])} if n=='P23' else seq
            if n=='P23': sq={**{15:'pyramid'},**sq}
            yp=[sq.get(pm[p-1],seq[pm[p-1]]) for p in pos[n]];a.append(np.mean(np.array(pred[n])==np.array(yp)))
        null.append(np.mean(a))
    p=(np.sum(np.array(null)>=acc)+1)/2001
    print(f'SHAPE TEST (leave-one-subject-out): accuracy {acc:.3f} | chance mean {np.mean(null):.3f} 95% {np.percentile(null,95):.3f} | p={p:.4f}')
    json.dump(dict(acc=acc,p=p,null95=float(np.percentile(null,95)),pred=pred,true=true),open('hmm_test_rel.json','w'),ensure_ascii=False)
