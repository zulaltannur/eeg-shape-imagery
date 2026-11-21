import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '07_things_eeg2_pretraining'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import open_clip,torch,numpy as np,glob,os
from PIL import Image
torch.set_num_threads(2)
m,_,pre=open_clip.create_model_and_transforms('ViT-B-32',pretrained='laion2b_s34b_b79k');m.eval()
for part in ['test_images','training_images']:
    fs=sorted(glob.glob(f'img/{part}/*/*.jpg'));E=[]
    with torch.no_grad():
        for i in range(0,len(fs),64):
            x=torch.stack([pre(Image.open(f).convert('RGB')) for f in fs[i:i+64]])
            e=m.encode_image(x);E.append((e/e.norm(dim=-1,keepdim=True)).numpy())
            if i%1280==0: print(part,i,flush=True)
    np.save(f'clip_{part}.npy',np.concatenate(E));open(f'clip_{part}.txt','w').write('\n'.join(fs))
print('DONE')
