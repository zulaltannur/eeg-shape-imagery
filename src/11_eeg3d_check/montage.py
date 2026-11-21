import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '11_eeg3d_check'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, mne, warnings; warnings.filterwarnings('ignore')
X=np.load('sub01/process_data_6s_100Hz.npy',mmap_mode='r'); X=np.asarray(X[::3])
C=np.corrcoef(X.transpose(1,0,2).reshape(64,-1)); np.fill_diagonal(C,0)
pos=mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
P={k.upper():v for k,v in pos.items()}
QUIK=['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']
QUIK_M=QUIK[:32]+['M1']+QUIK[32:41]+['M2']+QUIK[41:]
ACTI=['FP1','FZ','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1','PZ','P3','P7','O1','OZ','O2','P4','P8','TP10','CP6','CP2','CZ','C4','T8','FT10','FC6','FC2','F4','F8','FP2','AF7','AF3','AFZ','F1','F5','FT7','FC3','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POZ','PO4','PO8','P6','P2','CPZ','CP4','TP8','C6','C2','FC4','FT8','F6','AF8','AF4','F2','IZ']
TH=['FP1','FP2','AF7','AF3','AFZ','AF4','AF8','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT9','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','FT10','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP9','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','TP10','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO3','POZ','PO4','PO8','O1','OZ','O2']
P['CB1']=P.get('CB1',P['PO9'] if 'PO9' in P else P['O1']);P['CB2']=P.get('CB2',P['PO10'] if 'PO10' in P else P['O2'])
P['M1']=P.get('M1',P['TP9']);P['M2']=P.get('M2',P['TP10'])
def score(names):
    names=names[:64]; Q=np.array([P[n] for n in names]); D=np.linalg.norm(Q[:,None]-Q[None],axis=-1)
    K=np.exp(-(D/0.03)**2); np.fill_diagonal(K,0)
    return float((C*K).sum()/K.sum())
rng=np.random.default_rng(0)
null=[score(list(rng.permutation(TH))) for _ in range(200)]
for nm,L in [x for x in [('Quik-Cap (M1/M2 none)',QUIK),('Quik-Cap (M1/M2 present, first 64)',QUIK_M),('actiCAP 64',ACTI),('10-10 row order (THINGS type)',TH)] if len(x[1])==64]:
    print(f'{nm:35s} n={len(L)} fit={score(L):.3f}')
print('random layout: mean',round(np.mean(null),3),'highest',round(np.max(null),3))
# local refinement: swaps of neighbouring channels (simulated annealing)
cur=list(QUIK_M);s=score(cur);best=(s,list(cur));T=0.01
for it in range(20000):
    i,j=rng.integers(64,size=2)
    if i==j: continue
    cur[i],cur[j]=cur[j],cur[i];s2=score(cur)
    if s2>s or rng.random()<np.exp((s2-s)/T): s=s2
    else: cur[i],cur[j]=cur[j],cur[i]
    T*=0.9997
    if s>best[0]: best=(s,list(cur))
print('improved fit',round(best[0],3));diff=[(k,QUIK_M[k],best[1][k]) for k in range(64) if QUIK_M[k]!=best[1][k]];print('changed',len(diff),diff[:20])
