import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '07_things_eeg2_pretraining'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,torch,pickle,warnings;warnings.filterwarnings('ignore');torch.set_num_threads(2)
from train_enc import Enc
enc=Enc();enc.load_state_dict(torch.load('enc.pt'));enc.eval()
D=pickle.load(open('ours.pkl','rb'));OUT={}
with torch.no_grad():
    for n,d in D.items():
        for c,v in d.items():
            S=v['S'].astype(np.float32);m=S.mean(axis=(0,1,3),keepdims=True);s=S.std(axis=(0,1,3),keepdims=True);S=(S-m)/(s+1e-6)
            x=torch.tensor(S.reshape(-1,17,100))
            h=enc.p(enc.t(x[:,None]))                       # output of the frozen convolution layers (before the projection head)
            h1=enc.h[1](enc.h[0](h))                         # first layer of the head (after GELU)
            OUT.setdefault(n,{})[c]=dict(H=h.reshape(S.shape[0],S.shape[1],-1).mean(1).numpy(),H1=h1.reshape(S.shape[0],S.shape[1],-1).mean(1).numpy(),y=np.array(v['y']),pos=np.array(v['pos']))
print('dimensions',OUT['P15']['B1_img']['H'].shape,OUT['P15']['B1_img']['H1'].shape)
pickle.dump(OUT,open('../../data/derived/16_boosting/a5_feats.pkl','wb'))
