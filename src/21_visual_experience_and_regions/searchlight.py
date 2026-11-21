import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '21_visual_experience_and_regions'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Data-driven scan: shape decoding for every channel neighbourhood × every band (all people, leave-one-subject-out), max-statistic correction
import numpy as np,json,pickle,warnings,sys,mne,time;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
from sklearn.linear_model import LogisticRegression
CL=['cone','cube','sphere','pyramid','cylinder'];BANDS=['theta','alpha','beta','gamma']
OL=pickle.load(open('../../data/derived/06_features/feats.pkl','rb'))
info=mne.read_epochs('../../data/clean_epochs/epo2/P15_block2_imagery_motionless-epo.fif',preload=False).info
adj,names=mne.channels.find_ch_adjacency(info,'eeg');adj=adj.toarray();NB=[np.where(adj[i])[0] for i in range(64)]
cond=sys.argv[1];out_f=f'../../results/21_visual_experience_and_regions/sl_{cond}.json'
R={n:d[cond] for n,d in OL.items() if cond in d}
def prep(X):return (X-X.mean(0))/(X.std(0)+1e-9)
ns=sorted(R);X={n:prep(R[n]['X']) for n in ns};Y={n:np.array([CL.index(q) for q in R[n]['y']]) for n in ns};POS={n:np.array(R[n]['pos']) for n in ns}
erp=cond.endswith('erp')
cells=[]
if erp:   # 64 channels × 15 time bins (50 ms): every neighbourhood, all bins
    for c in range(64): cells.append((f'{names[c]}|ERP',[ch*15+t for ch in NB[c] for t in range(15)]))
else:
    for b in range(4):
        for c in range(64): cells.append((f'{names[c]}|{BANDS[b]}',[b*64+ch for ch in NB[c]]))
pred={}
T=time.time()
for name,ix in cells:
    pr={}
    for te in ns:
        tr=[m for m in ns if m!=te];clf=LogisticRegression(C=0.1,max_iter=2000).fit(np.vstack([X[m][:,ix] for m in tr]),np.concatenate([Y[m] for m in tr]))
        pr[te]=clf.predict(X[te][:,ix])
    pred[name]=pr
print('training done',round(time.time()-T),flush=True)
obs={k:np.mean([np.mean(v[n]==Y[n]) for n in ns]) for k,v in pred.items()}
rng=np.random.default_rng(0);mx=[];null_cell={k:[] for k in obs}
for _ in range(1000):
    pi=rng.permutation(15);Yp={}
    for n in ns:
        lab=np.full(15,-1);lab[POS[n]-1]=Y[n];q=lab[pi];q=q[q>=0];Yp[n]=q if len(q)==len(Y[n]) else rng.permutation(Y[n])
    a={k:np.mean([np.mean(v[n]==Yp[n]) for n in ns]) for k,v in pred.items()};mx.append(max(a.values()))
    for k in a: null_cell[k].append(a[k])
mx=np.array(mx)
keys=list(obs);NM=np.array([null_cell[k] for k in keys]).T;O=np.array([obs[k] for k in keys]);thr=np.percentile(NM,95,axis=0)
grp=[k.split('|')[1] for k in keys];chi=[names.index(k.split('|')[0]) for k in keys]
def clusters(vals):
    sup=vals>thr;cl=[];seen=set()
    for i in np.where(sup)[0]:
        if i in seen: continue
        st=[i];comp=[];seen.add(i)
        while st:
            u=st.pop();comp.append(u)
            for v in np.where(sup)[0]:
                if v not in seen and grp[v]==grp[u] and adj[chi[u],chi[v]]: seen.add(v);st.append(v)
        cl.append((float(np.sum(vals[comp]-0.2)),comp))
    return cl
CLO=clusters(O);cmax=np.array([max([m for m,_ in clusters(NM[i])],default=0) for i in range(len(NM))])
cluster=[dict(band_name=grp[c[0]],channels=[names[chi[j]] for j in c],mass=round(m,4),mean_accuracy=round(float(np.mean(O[c])),4),p_cluster=float((1+np.sum(cmax>=m))/(1+len(cmax)))) for m,c in sorted(CLO,key=lambda x:-x[0])]
res={k:dict(accuracy=round(float(o),4),p_cell=float((1+np.sum(np.array(null_cell[k])>=o))/1001),p_corrected=float((1+np.sum(mx>=o))/1001)) for k,o in obs.items()}
json.dump(dict(n=len(ns),cell=res,clusters=cluster),open(out_f,'w'),ensure_ascii=False)
top=sorted(res.items(),key=lambda x:-x[1]['accuracy'])[:6]
print(cond,len(ns),[(x['band_name'],x['channels'][:8],len(x['channels']),x['mean_accuracy'],round(x['p_cluster'],3)) for x in cluster[:3]]);print([(k,v['accuracy'],round(v['p_cell'],3),round(v['p_corrected'],3)) for k,v in top],flush=True)
