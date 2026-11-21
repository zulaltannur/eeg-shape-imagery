import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '20_time_frequency'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Block 2: channel × band power for clean 1-s segments (for time and frequency analysis)
import mne,numpy as np,glob,pickle,warnings;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
SF=256;exec(open('../../src/05_participant_profiles/battery.py').read().split('R = {}')[0])
FB={'theta':(4,7),'alpha':(8,13),'beta':(14,25),'gamma':(26,40)};CLEAN='../../data'
D={}
for n in sorted(G):
    f=[x for x in glob.glob(f'{CLEAN}/clean_epochs/epo2/{n}_block2*_touch_motionless-epo.fif') if 'nocube' not in x and 'third' not in x]
    if not f: continue
    for cond,ff in [('B2_touch',f[0]),('B2_img',f[0].replace('touch','imagery'))]:
        ep=mne.read_epochs(ff,preload=True);md=ep.metadata.reset_index(drop=True);X=ep.copy().set_eeg_reference('average').get_data()
        segs=list(X);m=chunk_masks(segs);nc=min(s.shape[1]//SF for s in segs)
        P=np.full((len(segs),nc,64,4),np.nan)
        for i,(s,mm) in enumerate(zip(segs,m)):
            for c in range(nc):
                if c<len(mm) and mm[c]:
                    p,fr=mne.time_frequency.psd_array_welch(s[:,c*SF:(c+1)*SF],SF,fmin=2,fmax=40,n_fft=SF,verbose=False)
                    P[i,c]=np.stack([p[:,(fr>=lo)&(fr<=hi)].mean(1) for lo,hi in FB.values()],1)
        y=np.array(md['shape']);pos=np.array(md.trial)
        if n=='P23': k=pos!=15;P,y,pos=P[k],y[k],pos[k]
        D.setdefault(n,{})[cond]=dict(P=P,y=y,pos=pos)
    print(n,D[n]['B2_img']['P'].shape,flush=True)
pickle.dump(D,open('../../results/20_time_frequency/tf.pkl','wb'))
