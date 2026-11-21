import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '16_accuracy_boosting'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np,json,sys
CL=['cone','cube','sphere','pyramid','cylinder']
def sig(res,name,nper=2000,seed=0):
    rng=np.random.default_rng(seed);seq={}
    for n,v in res.items():
        if not(n=='P23' and 'Block 2' in name):
            for p,y in zip(v['pos'],v['y']): seq.setdefault(p,y)
    pred={n:np.array(CL)[np.array(v['P']).argmax(1)] for n,v in res.items()};Y={n:np.array(v['y']) for n,v in res.items()};pos={n:v['pos'] for n,v in res.items()}
    acc=np.mean([np.mean(pred[n]==Y[n]) for n in pred]);null=[]
    for _ in range(nper):
        pm=rng.permutation(np.arange(1,16));a=[]
        for n in pred:
            own=dict(zip(pos[n],Y[n]))
            lab=[(own.get(pm[p-1],'pyramid') if (n=='P23' and 'Block 2' in name) else seq.get(pm[p-1],Y[n][i])) for i,p in enumerate(pos[n])]
            a.append(np.mean(pred[n]==np.array(lab)))
        null.append(np.mean(a))
    return acc,(np.sum(np.array(null)>=acc)+1)/(nper+1),np.percentile(null,95)
if __name__=='__main__':
    r=json.load(open(sys.argv[1]));out={}
    for name,res in r.items():
        a,p,n95=sig(res,name);out[name]=dict(acc=a,p=p,null95=n95);print(f'{name:32s} %{a*100:.1f}  chance 95% {n95*100:.1f}  p={p:.4f}',flush=True)
    json.dump(out,open(sys.argv[1].replace('.json','_sig.json'),'w'),ensure_ascii=False)
