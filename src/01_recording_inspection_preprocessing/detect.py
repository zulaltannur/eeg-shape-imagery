import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '01_recording_inspection_preprocessing'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, json, glob
mne.set_log_level('ERROR')
S1=[0,12,23,33,43,50,64,71,80,90,100,112,123,135,143]; D1=[2.5,2.5,1,1,1,2.5,1,1,1,1,1,1,1,1,2.5]
TOUCH=[0,29,57,84,112,139,166,195,222,251,279,306,335,363,391]; IMG=[13,42,70,97,125,152,179,208,235,264,292,319,348,376,404]
S2=sorted(TOUCH+IMG); D2=[3]*30
S3=[0,16,33,51,67,85,102,119,135,153,169,187,203,221,238]; D3=[5]*15  # visual: null for auditory
FC=['Fz','FCz','Cz','FC1','FC2']; INF=['P9','P10','TP7','TP8']
SF=256
def template(dur):
    t=np.arange(0,dur+0.45,1/SF)
    g=lambda c,w,a: a*np.exp(-0.5*((t-c)/w)**2)
    return g(0.10,0.025,-1)+g(0.19,0.035,0.8)+g(dur+0.10,0.025,-0.5)+g(dur+0.19,0.035,0.4)
def score_series(x,tpl):
    n=len(tpl); tz=(tpl-tpl.mean())/tpl.std()
    from numpy.lib.stride_tricks import sliding_window_view
    W=sliding_window_view(x,n)
    Wz=(W-W.mean(1,keepdims=True))/(W.std(1,keepdims=True)+1e-12)
    r=Wz@tz/n
    amp=W.std(1)
    return r,amp
def run(name,ons,durs):
    j=json.load(open(f'../../data/preprocessing_proc/{name}.json')); on=[t for t,a,b in j['transitions'] if b==49]
    if not on: return None
    t0=on[0]; raw=mne.io.read_raw_fif(f'../../data/preprocessing_proc/{name}_raw.fif',preload=True); raw.filter(1,15)
    fc=[c for c in FC if c not in raw.info['bads']]; inf=[c for c in INF if c not in raw.info['bads']]
    x=(raw.get_data(picks=fc).mean(0)-raw.get_data(picks=inf).mean(0))*1e6
    res=[]
    cache={}
    for k,(p,d) in enumerate(zip(ons,durs)):
        if d not in cache: cache[d]=score_series(x,template(d))
        r,amp=cache[d]
        nxt=ons[k+1] if k+1<len(ons) else p+12
        a=int((t0+p-0.5)*SF); b=int((t0+min(nxt-0.5,p+8))*SF)
        b=min(b,len(r)); 
        if a<0 or b<=a: res.append(None); continue
        i=a+int(np.argmax(r[a:b])); res.append((round(i/SF-t0-p,3),round(float(r[i]),3)))
    # background distribution: whole recording max-in-window for comparable window widths
    return res
out={}
for blk,ons,durs,pat in [('block1',S1,D1,'../../data/preprocessing_proc/*_block1.json'),('block2',S2,D2,'../../data/preprocessing_proc/*_block2*.json'),('block3_null',S3,D3,'../../data/preprocessing_proc/*_block3.json')]:
    for f in sorted(glob.glob(pat)):
        n=f.split('/')[-1][:-5]
        if 'error' in json.load(open(f)): continue
        r=run(n,ons,durs)
        if r: out[n]=(blk,r)
json.dump(out,open('detect.json','w'))
import collections
for blk in ['block1','block2','block3_null']:
    sc=[s for n,(b,r) in out.items() if b==blk for x in r if x for s in [x[1]]]
    print(blk,'n',len(sc),'score pct 25/50/75/90',np.round(np.percentile(sc,[25,50,75,90]),3))
