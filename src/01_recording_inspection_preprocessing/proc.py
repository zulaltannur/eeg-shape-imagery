import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '01_recording_inspection_preprocessing'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, json, glob, os, sys
mne.set_log_level('ERROR')
B64 = mne.channels.make_standard_montage('biosemi64').ch_names
src=sys.argv[1]
m128=mne.channels.make_standard_montage('biosemi128').get_positions()
_pairs=json.load(open('map64to128.json'))
pos={B64[i]:m128['ch_pos'][b] for i,(a,b) in enumerate(_pairs)}
MONT=mne.channels.make_dig_montage(ch_pos=pos,nasion=m128['nasion'],lpa=m128['lpa'],rpa=m128['rpa'],coord_frame='head')
for f in sorted(glob.glob(os.path.join(src,'*.bdf'))):
    name=os.path.basename(f)[:-4]
    if os.path.exists(f'../../data/preprocessing_proc/{name}.json'): continue
    try:
        raw=mne.io.read_raw_bdf(f,preload=True)
    except Exception as e:
        json.dump({'name':name,'error':str(e)},open(f'../../data/preprocessing_proc/{name}.json','w')); print(name,'ERR',e); continue
    sf=raw.info['sfreq']; n_orig=len(raw.ch_names)
    st=raw.get_data(picks=['Status'])[0].astype(np.int64)&0xFFFF
    idx=np.nonzero(np.diff(st))[0]+1
    trans=[[round(i/sf,4),int(st[i-1]),int(st[i])] for i in idx]
    AB=[f'A{i}' for i in range(1,33)]+[f'B{i}' for i in range(1,33)]
    pairs=json.load(open('map64to128.json'))
    layout='A-B banks (64 electrodes), 128-channel cap positions'
    extra=[c for c in raw.ch_names if c not in AB]
    raw.drop_channels(extra); raw.reorder_channels(AB); raw.rename_channels(dict(zip(raw.ch_names,B64))); raw.set_montage(MONT)
    raw.filter(0.5,30); raw.resample(256)
    sd=raw.get_data().std(1); med=np.median(sd)
    bads=[raw.ch_names[i] for i in np.nonzero((sd>3*med)|(sd<med/5))[0]]
    raw.info['bads']=bads; raw.set_eeg_reference('average')
    raw.save(f'../../data/preprocessing_proc/{name}_raw.fif',overwrite=True)
    info={'name':name,'sfreq':sf,'dur':round(raw.times[-1],2),'meas_date':str(raw.info['meas_date']),'transitions':trans[:200],'n_trans':len(trans),'bads':bads,'extra_channels':extra,'layout':layout,'n_ch_orig':n_orig,'median_sd_uV':round(float(med)*1e6,2)}
    json.dump(info,open(f'../../data/preprocessing_proc/{name}.json','w'),ensure_ascii=False)
    print(name,info['dur'],'trans',len(trans),trans[:4],'bads',bads)
