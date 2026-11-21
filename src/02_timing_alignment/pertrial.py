import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '02_timing_alignment'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, json, glob, os, pandas as pd
from scipy.ndimage import uniform_filter1d
mne.set_log_level('ERROR')
AB=[f'A{i}' for i in range(1,33)]+[f'B{i}' for i in range(1,33)]
TOUCH=[0,29,57,84,112,139,166,195,222,251,279,306,335,363,391]; IMG=[13,42,70,97,125,152,179,208,235,264,292,319,348,376,404]
L2=['pyramid','cone','sphere','cylinder','cube','cone','sphere','cylinder','pyramid','cube','sphere','pyramid','cylinder','cube','cone']
R=0.25  # s bin
rows=[]; summary=[]
srcs=sorted(glob.glob('../../data/raw_bdf/*_block2*.bdf'))
for f in srcs:
    n=os.path.basename(f)[:-4]
    rep=json.load(open(f'../../results/03_cleaning_epoching/{n}.json')); t0=rep['used_onset_s']
    raw=mne.io.read_raw_bdf(f,preload=True); raw.pick(AB); raw.resample(128); lo=raw.filter(0.3,3).get_data()
    w=int(R*128); nb=lo.shape[1]//w
    env=np.log(np.median(np.abs(lo[:,:nb*w]).reshape(64,nb,w).mean(2),0)); env=uniform_filter1d(env,3)
    z=(env-np.median(env))/(1.4826*np.median(np.abs(env-np.median(env)))+1e-9)
    tt=np.arange(nb)*R+R/2
    for k,(d,i,s) in enumerate(zip(TOUCH,IMG,L2)):
        if 'nocube' in n and s=='cube': continue
        D=t0+d
        def peak(a,b,default):
            m=(tt>=D+a)&(tt<=D+b)
            if m.sum()<4: return D+default,np.nan,'none'
            j=np.argmax(z[m]); pz=z[m][j]
            return tt[m][j], pz, ('trace' if pz>=2 else 'mean')
        take,tz,ts=peak(-5,3,-2); ret,rz,rs=peak(7,13,11)
        if ts=='mean': take=D-2
        if rs=='mean': ret=D+11
        if ts=='none' and rs=='none': continue
        rows.append(dict(file=n,participant=n.split('_')[0],trial=k+1,shape=s,plan_touch_beep_s=round(D,2),taking_s=round(take,2),taking_source=ts,taking_z=round(float(tz),1) if tz==tz else None,
                         back_giving_s=round(ret,2),back_source=rs,back_z=round(float(rz),1) if rz==rz else None,exploration_duration_s=round(ret-take,2),plan_imagine_beep_s=round(t0+i,2)))
df=pd.DataFrame(rows); df.to_csv('pertrial.csv',index=False)
print(len(df),'trial'); print(df[['taking_source','back_source']].apply(pd.Series.value_counts))
d=df[(df.taking_source=='trace')]; print('taking - touch beep (s): median',round((d.taking_s-d.plan_touch_beep_s).median(),2),'IQR',np.round(np.percentile(d.taking_s-d.plan_touch_beep_s,[25,75]),2))
d=df[(df.back_source=='trace')]; print('back - touch beep: median',round((d.back_giving_s-d.plan_touch_beep_s).median(),2),'IQR',np.round(np.percentile(d.back_giving_s-d.plan_touch_beep_s,[25,75]),2))
print('exploration duration median',df.exploration_duration_s.median(),'IQR',np.percentile(df.exploration_duration_s,[25,75]))
print(df.groupby('shape').exploration_duration_s.median())
