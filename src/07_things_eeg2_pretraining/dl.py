import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '07_things_eeg2_pretraining'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json,urllib.request,subprocess,os,zipfile,numpy as np,glob,shutil
def files(node):
    d=json.load(urllib.request.urlopen(f'https://api.osf.io/v2/nodes/{node}/files/osfstorage/'))
    return {x['attributes']['name']:x['id'] for x in d['data']}
def get(node,fid,out):
    subprocess.run(['curl','-sS','-L','--retry','3','-o',out,f'https://files.osf.io/v1/resources/{node}/providers/osfstorage/{fid}?action=download'],check=True)
im=files('y63gw')
for f in ['image_metadata.npy','training_images.zip']:
    if not os.path.exists(f): get('y63gw',im[f],f); print('downloaded',f,os.path.getsize(f),flush=True)
pp=files('anp5v')
for s in ['sub-07','sub-08','sub-09','sub-10']:
    if os.path.exists(f'{s}_train.npy'): continue
    get('anp5v',pp[s+'.zip'],'tmp.zip')
    z=zipfile.ZipFile('tmp.zip');names=z.namelist();print(s,names,flush=True)
    for nm in names:
        if nm.endswith('.npy'):
            z.extract(nm,'tmp');d=np.load(os.path.join('tmp',nm),allow_pickle=True).item()
            X=d['preprocessed_eeg_data'].astype(np.float16)
            tag='train' if 'training' in nm else 'test'
            np.save(f'{s}_{tag}.npy',X)
            if s=='sub-01': json.dump({'ch_names':list(d['ch_names']),'times':[float(t) for t in d['times']]},open('meta.json','w'))
            print(' ',tag,X.shape,flush=True)
    shutil.rmtree('tmp');os.remove('tmp.zip')
print('DONE')
