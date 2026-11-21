import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '01_recording_inspection_preprocessing'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import mne, numpy as np, sys
mne.set_log_level('ERROR')
for f in sys.argv[1:]:
    raw = mne.io.read_raw_bdf(f, preload=False)
    print('=====', f.split('/')[-1])
    print('sfreq', raw.info['sfreq'], 'n_ch', len(raw.ch_names), 'dur_s', round(raw.times[-1],2))
    print('meas_date', raw.info['meas_date'])
    print('channels', raw.ch_names)
    st = raw.get_data(picks=['Status'])[0].astype(np.int64)
    low = st & 0xFFFF
    vals, cnt = np.unique(low, return_counts=True)
    print('status low16 unique:', list(zip(vals.tolist(), cnt.tolist()))[:20])
    hi = (st >> 16) & 0xFF
    v2,c2=np.unique(hi,return_counts=True); print('status hi bits:', list(zip(v2.tolist(),c2.tolist()))[:10])
    d = np.diff(low); idx = np.nonzero(d)[0]
    print('n transitions', len(idx))
    for i in idx[:60]:
        print(f'  t={ (i+1)/raw.info["sfreq"]:.3f}s  {low[i]} -> {low[i+1]}')
