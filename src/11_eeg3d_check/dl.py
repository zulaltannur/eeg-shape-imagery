import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '11_eeg3d_check'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import subprocess,zipfile,numpy as np,os
for i in range(1,13):
    s=f'sub{i:02d}'
    if os.path.exists(f'{s}_1s.npy'): continue
    if not os.path.exists(f'{s}.zip'):
        subprocess.run(['curl','-sS','-L','--retry','3','-o',f'{s}.zip',f'https://huggingface.co/datasets/guozq21/neuro-3D/resolve/main/EEGdata/{s}.zip'],check=True)
    z=zipfile.ZipFile(f'{s}.zip');z.extract(f'{s}/process_data_1s_250Hz.npy','tmp');z.extract(f'{s}/name_label.npy','tmp')
    np.save(f'{s}_1s.npy',np.load(f'tmp/{s}/process_data_1s_250Hz.npy').astype(np.float32));np.save(f'{s}_lab.npy',np.load(f'tmp/{s}/name_label.npy'))
    os.remove(f'tmp/{s}/process_data_1s_250Hz.npy');os.remove(f'{s}.zip');print('ready',s,flush=True)
print('DONE')
