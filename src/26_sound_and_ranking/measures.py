import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '26_sound_and_ranking'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# All measures per person
import numpy as np,json,pickle,os,warnings,itertools,pandas as pd,mne;warnings.filterwarnings('ignore');mne.set_log_level('ERROR')
CL=['cone','cube','sphere','pyramid','cylinder'];R=json.load(open('../../results/16_accuracy_boosting/final_res.json'))
CLV={'P01','P15','P02','P13','P17','P06','P03','P10','P23'};LVL={'P18','P21','P16','P20'}
k=pd.read_excel('../../participants/participant_information.xlsx');G0=dict(zip(k['code'],k['analysis_group']))
G=lambda n:'Congenitally limited' if n in CLV else ('Typical vision then lost' if n in LVL else 'Sighted')
ALL=sorted(G0)
M={n:{} for n in ALL}
EG=np.array([1,0,1,0,1]);SV=np.array([1,0,0,1,0])
for w,lab in [('Block 1 · word onset','Readability · B1 word onset'),('Block 1 · imagery','Readability · B1 imagery from hearing'),('Block 2 · touch','Readability · B2 touch'),('Block 2 · imagery','Readability · B2 imagery from touch'),('Block 2 · touch to imagery','Readability · B2 touch to imagery'),('Block 3 · picture onset','Readability · B3 picture onset'),('Block 3 · picture viewing','Readability · B3 picture viewing'),('Block 3 · imagery','Readability · B3 imagery from seeing')]:
    for n,v in R[w].items():
        P=np.array(v['P']);y=np.array([CL.index(q) for q in v['y']]);M[n][lab]=float(np.mean(P.argmax(1)==y))
        if w in ('Block 1 · imagery','Block 2 · imagery','Block 3 · imagery'):
            b=w.split(' · ')[0].replace('Block ','B')
            M[n][f'Curvature readability · {b} imagery']=float(np.mean(((P@EG)>0.6)==(EG[y]==1)))
            M[n][f'Pointedness readability · {b} imagery']=float(np.mean(((P@SV)>0.4)==(SV[y]==1)))
V=json.load(open('../../results/23_three_groups_cases_connectivity/case_res.json'))['person']
for n,v in V.items(): M[n]['Pattern closeness · B2 (+ congenitally limited, − sighted)']=v['closeness']
KR=json.load(open('../../results/25_stability_and_P04/stability_res.json'))
for w,lab in [('Block 1 · imagery','Stability · B1 imagery'),('Block 2 · imagery','Stability · B2 imagery'),('Block 3 · imagery','Stability · B3 imagery')]:
    for n,(s,a) in KR[w]['person'].items(): M[n][lab]=s
Z=json.load(open('../../results/23_three_groups_cases_connectivity/connectivity_res.json'))
for c,lab in [('B2_touch','Visual (occipital) region activity · during touch'),('B2_img','Visual (occipital) region activity · during imagery from touch')]:
    for n,v in Z['occipital_second'][c].items(): M[n][lab]=-float(np.mean(v[:7]))
for n,v in Z['person'].items():
    M[n]['Touch–occipital synchrony · alpha (during touch)']=np.mean([v['touch|alpha|touch left↔occipital'],v['touch|alpha|touch right↔occipital']])
    M[n]['Touch–occipital synchrony · beta (during touch)']=np.mean([v['touch|beta|touch left↔occipital'],v['touch|beta|touch right↔occipital']])
TF=pickle.load(open('../../results/20_time_frequency/tf.pkl','rb'));names=mne.read_epochs('../../data/clean_epochs/epo2/P15_block2_imagery_motionless-epo.fif',preload=False).ch_names
cen=[names.index(c) for c in 'C3 CP3 C1 C5 CP5 C4 CP4 C2 C6 CP6'.split()]
for n,d in TF.items():
    L=lambda c,b:np.log(np.nanmean(d[c]['P'].reshape(-1,64,4),0))[:,b]
    M[n]['Hand-touch region activity · during touch (low central alpha)']=-float(L('B2_touch',1)[cen].mean()-L('B2_touch',1).mean())
    M[n]['Alpha increase on moving to imagery (whole head)']=float((L('B2_img',1)-L('B2_touch',1)).mean())
    M[n]['Beta increase on moving to imagery (whole head)']=float((L('B2_img',2)-L('B2_touch',2)).mean())
    M[n]['Recording quality · B2 clean-second ratio']=float(np.mean([np.mean(~np.isnan(d[c]['P'][:,:,0,0])) for c in ['B2_touch','B2_img']]))
exec(open('../../src/21_visual_experience_and_regions/mantel.py').read().split('def feats')[0].split("TF=pickle")[0]+"\n")
VIS={'P07':4,'P11':4,'P20':0,'P08':4,'P01':2,'P09':4,'P22':4,'P03':2,'P18':2,'P14':4,'P21':2,'P19':4,'P13':1,'P17':1,'P16':1,'P05':4,'P15':2,'P06':1,'P02':2,'P10':0,'P12':4,'P04':4,'P23':0}
AGE=dict(zip(k['code'],k['age']));YRS={n:(AGE[n] if G0[n]=='Sighted' else 0) for n in AGE};YRS.update({'P20':22,'P18':15,'P21':38,'P16':14,'P23':0,'P13':0})
for n in ALL: M[n]['Person · years with typical vision']=YRS[n];M[n]['Person · residual vision level (0–4)']=VIS[n]
pickle.dump(dict(M=M,G={n:G(n) for n in ALL}),open('measure.pkl','wb'));print(pd.DataFrame(M).T.shape)
