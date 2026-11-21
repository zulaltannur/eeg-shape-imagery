import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '01_recording_inspection_preprocessing'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, mne
exec(open('../../src/01_recording_inspection_preprocessing/erp2.py').read().split('res={}')[0])
raw1,t01=load('../../data/raw_bdf/P23_block1.bdf')
raw2,t02=load('../../data/raw_bdf/P23_block2_trial1.bdf')
s1=[0,12,23,33,43,50,64,71,80,90,100,112,123,135,143]
touch=[0,29,57,84,112,139,166,195,222,251,279,306,335,363,391]; img=[13,42,70,97,125,152,179,208,235,264,292,319,348,376,404]
rng=np.random.default_rng(2)
def scan(raw,t0,ons):
    ep=epochs(raw,[t0+x-0.2 for x in ons]); # wide window
    sf=raw.info['sfreq']
    ev=np.array([[int(round((t0+x)*sf)),0,1] for x in ons])
    ep=mne.Epochs(raw,ev,tmin=-0.3,tmax=1.2,baseline=None,reject=dict(eeg=200e-6),preload=True,picks='eeg')
    chs=[c for c in ('FCz','Cz','Fz','FC1','FC2') if c not in raw.info['bads']]
    m=ep.copy().pick(chs).get_data().mean(1).mean(0)*1e6; t=ep.times
    out=[]
    for L in np.arange(0,0.61,0.01):
        b=m[(t>=L-0.1)&(t<L)].mean()
        n1=m[(t>=L+0.08)&(t<L+0.14)].mean()-b
        p2=m[(t>=L+0.16)&(t<L+0.26)].mean()-b
        out.append((round(L,2),round(p2-n1,2)))
    return out,len(ep)
for name,raw,t0,ons in [('B1',raw1,t01,s1),('B2',raw2,t02,sorted(touch+img))]:
    o,n=scan(raw,t0,ons); best=max(o,key=lambda z:z[1])
    # null: same scan on random onset sets
    nl=[]
    for k in range(150):
        rr=list(np.sort(rng.choice(np.arange(3,raw.times[-1]-3-t0,0.05),len(ons),replace=False)))
        oo,_=scan(raw,t0,rr); nl.append(max(z[1] for z in oo))
    p=(np.sum(np.array(nl)>=best[1])+1)/151
    print(name,'kept',n,'best lag',best,'p',round(p,3))
