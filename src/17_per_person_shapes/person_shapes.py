import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '17_per_person_shapes'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Shapes generated from EEG for each participant: table + individual images
import numpy as np,json,sys,os,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
sys.path.insert(0,'../../src/09_superquadric_bridge');from sq import PAR,draw
CL=['cone','cube','sphere','pyramid','cylinder'];PR=np.array([PAR[c] for c in CL])
R=json.load(open('../../results/16_accuracy_boosting/final_res.json'))
OUT='../../results/17_per_person_shapes/person_shapes'
BL={'block1_hearing':[('Block 1 · word onset','1_word_onset','Block 1 · moment of hearing the word'),('Block 1 · imagery','2_imagery','Block 1 · imagining what was heard')],
    'block2_touch':[('Block 2 · touch','1_touch','Block 2 · touch'),('Block 2 · imagery','2_imagery','Block 2 · imagining what was touched'),('Block 2 · touch to imagery','3_touch_to_imagery','Block 2 · touch-to-imagery transfer')],
    'block3_seeing':[('Block 3 · picture onset','1_picture_onset','Block 3 · moment of seeing the picture'),('Block 3 · picture viewing','2_picture_viewing','Block 3 · picture viewing'),('Block 3 · imagery','3_imagery','Block 3 · imagining what was seen')]}
def soft(P,T):L=np.log(P+1e-9)/T;E=np.exp(L-L.max(1,keepdims=True));return E/E.sum(1,keepdims=True)
cal={}
for w,res in R.items():
    for n in res:   # temperature from the other people only
        P=np.vstack([res[m]['P'] for m in res if m!=n]);Y=np.concatenate([[CL.index(y) for y in res[m]['y']] for m in res if m!=n])
        Ts=np.linspace(0.05,1.5,59);T=Ts[int(np.argmax([np.log(soft(P,T)[np.arange(len(Y)),Y]+1e-12).mean() for T in Ts]))]
        cal[(w,n)]=soft(np.array(res[n]['P']),T)
people=sorted({n for res in R.values() for n in res})
only=sys.argv[1:] or people
def single(p,path,title):
    if os.path.exists(path): return
    fig=plt.figure(figsize=(3.2,3.6));draw(fig.add_subplot(111,projection='3d'),p);fig.suptitle(title,fontsize=8.5,y=0.97)
    fig.savefig(path,dpi=100,bbox_inches='tight');plt.close(fig)
summary={}
for n in only:
    table_rows=[]
    for bk,ws in BL.items():
        for w,ad,label in ws:
            if n not in R[w]: continue
            Pc=cal[(w,n)];y=np.array(R[w][n]['y']);pos=R[w][n]['pos'];d=os.path.join(OUT,n,bk,ad);os.makedirs(d,exist_ok=True)
            correct=(np.array(CL)[Pc.argmax(1)]==y);acc=correct.mean();summary.setdefault(n,{})[w]=round(float(acc),3)
            cell=[]
            for c in CL:
                idx=np.where(y==c)[0]
                if len(idx)==0: cell.append(None);continue
                pm=(Pc[idx]@PR).mean(0);pr=Pc[idx].mean(0);guess=CL[int(pr.argmax())]
                single(pm,os.path.join(d,f'{c}_mean.png'),f'{n} · {label}\ntrue: {c} ({len(idx)} trial mean)\nEEG estimate: {guess} %{pr.max()*100:.0f} · {int(correct[idx].sum())}/{len(idx)} correct')
                for k,i in enumerate(idx):
                    tk=CL[int(Pc[i].argmax())]
                    single(Pc[i]@PR,os.path.join(d,f'{c}_trial{k+1}.png'),f'{n} · {label}\ntrue: {c} · trial {k+1} (order {pos[i]})\nEEG estimate: {tk} %{Pc[i].max()*100:.0f} {"✓" if tk==c else "✗"}')
                cell.append((pm,f'{guess} %{pr.max()*100:.0f}\n{int(correct[idx].sum())}/{len(idx)} correct'))
            table_rows.append((label,acc,cell))
    path=os.path.join(OUT,n,f'{n}_table.png')
    fig=plt.figure(figsize=(2.0*6,2.15*(len(table_rows)+1)+0.8));C=6
    ax=fig.add_subplot(len(table_rows)+1,C,1);ax.axis('off');ax.text(0.5,0.5,'TRUE SHAPE\n(CAD, 60 mm)',ha='center',va='center',fontsize=10,weight='bold')
    for j,c in enumerate(CL): draw(fig.add_subplot(len(table_rows)+1,C,j+2,projection='3d'),PAR[c],c)
    for i,(et,acc,hc) in enumerate(table_rows):
        ax=fig.add_subplot(len(table_rows)+1,C,(i+1)*C+1);ax.axis('off');ax.text(0.5,0.5,et.replace(' · ','\n')+f'\naccuracy %{acc*100:.0f}',ha='center',va='center',fontsize=9)
        for j,h in enumerate(hc):
            a=fig.add_subplot(len(table_rows)+1,C,(i+1)*C+j+2,projection='3d')
            if h is None: a.set_axis_off();continue
            draw(a,h[0],h[1])
    fig.suptitle(f'{n} · Shapes generated from EEG (the model never saw this person; mean of trials per shape)',fontsize=11)
    fig.savefig(path,dpi=90,bbox_inches='tight');plt.close(fig)
    print(n,summary[n],flush=True)
json.dump(summary,open(f'../../results/17_per_person_shapes/summary_{"_".join(only) if len(only)<3 else 'all'}.json','w'),ensure_ascii=False)
