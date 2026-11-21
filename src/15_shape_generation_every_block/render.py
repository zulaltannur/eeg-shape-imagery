import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '15_shape_generation_every_block'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,json,sys,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
sys.path.insert(0,'../../src/09_superquadric_bridge');from sq import PAR,draw
CL=['cone','cube','sphere','pyramid','cylinder'];PR=np.array([PAR[c] for c in CL])
r=json.load(open('gen_res.json'));ev=json.load(open('gen_eval.json'))
order=['Block 1 · word onset','Block 1 · imagery','Block 2 · touch','Block 2 · imagery','Block 2 · touch to imagery','Block 3 · picture onset','Block 3 · picture viewing','Block 3 · imagery']
def soft(P,T):L=np.log(P+1e-9)/T;E=np.exp(L-L.max(1,keepdims=True));return E/E.sum(1,keepdims=True)
cal={};Tsel={}
for name in order:
    res=r[name];names=list(res)
    for n in names:  # temperature: only with the prediction-label pairs of the other people (maximum log-likelihood)
        P=np.vstack([res[m]['P'] for m in names if m!=n]);Y=np.concatenate([[CL.index(y) for y in res[m]['y']] for m in names if m!=n])
        Ts=np.linspace(0.05,1.5,59);ll=[np.log(soft(P,T)[np.arange(len(Y)),Y]+1e-12).mean() for T in Ts];T=Ts[int(np.argmax(ll))]
        Tsel.setdefault(name,[]).append(T);cal[(name,n)]=soft(np.array(res[n]['P']),T)
fig=plt.figure(figsize=(2.05*(len(order)+1),2.25*5+1.6))
for i,c in enumerate(CL):
    draw(fig.add_subplot(5,len(order)+1,i*(len(order)+1)+1,projection='3d'),PAR[c],('TRUE SHAPE\n(CAD, 60 mm)\n' if i==0 else '')+c)
    for j,name in enumerate(order):
        res=r[name];Ps=np.vstack([cal[(name,n)][np.array(res[n]['y'])==c] for n in res if (np.array(res[n]['y'])==c).any()])
        p=(Ps@PR).mean(0);e=ev[name]
        sig='★ ' if e['p_acc']<.05 else ''
        t=(sig+name.replace(' · ','\n')+f"\n%{e['acc']*100:.1f} (p={e['p_acc']:.3f})\n" if i==0 else '')+f"{c} trials"
        draw(fig.add_subplot(5,len(order)+1,i*(len(order)+1)+j+2,projection='3d'),p,t)
fig.suptitle('Shapes generated from EEG for every block (in never-seen people, calibrated probability-weighted superquadric average)\n★ = accuracy significantly above chance (same-order permutation test)',fontsize=11)
plt.savefig('shapes_from_every_block.png',dpi=95,bbox_inches='tight')
print({k:round(float(np.median(v)),2) for k,v in Tsel.items()})
json.dump({f'{a}|{b}':v.tolist() for (a,b),v in cal.items()},open('gen_cal.json','w'),ensure_ascii=False)
