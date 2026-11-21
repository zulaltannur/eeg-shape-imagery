import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
CL=['cone','cube','sphere','pyramid','cylinder']
# parameters: e1 (vertical cornerness: 0 cornered, 1 round), e2 (horizontal), tau (tapering 0-1)
PAR={'sphere':(1.0,1.0,0.0),'cube':(0.1,0.1,0.0),'cylinder':(0.1,1.0,0.0),'cone':(0.1,1.0,1.0),'pyramid':(0.1,0.1,1.0)}
def fexp(x,e): return np.sign(x)*np.abs(x)**e
def surface(e1,e2,tau,n=70):
    e1=max(e1,0.06);e2=max(e2,0.06)
    eta=np.linspace(-np.pi/2,np.pi/2,n);om=np.linspace(-np.pi,np.pi,n);E,W=np.meshgrid(eta,om)
    x=fexp(np.cos(E),e1)*fexp(np.cos(W),e2);y=fexp(np.cos(E),e1)*fexp(np.sin(W),e2);z=fexp(np.sin(E),e1)
    k=1-np.clip(tau,0,1)*(z+1)/2; return x*k*0.5,y*k*0.5,z*0.5   # 60 mm -> unit 1
LS=LightSource(azdeg=315,altdeg=40)
def draw(ax,p,title=None,alpha=1.0):
    X,Y,Z=surface(*p);ax.plot_surface(X,Y,Z,color=(0.72,0.12,0.10),shade=True,lightsource=LS,linewidth=0,antialiased=False,alpha=alpha)
    ax.set_box_aspect((1,1,1));L=0.55;ax.set_xlim(-L,L);ax.set_ylim(-L,L);ax.set_zlim(-L,L);ax.view_init(18,35);ax.set_axis_off()
    if title: ax.set_title(title,fontsize=9)
if __name__ == '__main__':
    fig=plt.figure(figsize=(12,2.8))
    for i,c in enumerate(CL): draw(fig.add_subplot(1,6,i+1,projection='3d'),PAR[c],c)
    draw(fig.add_subplot(1,6,6,projection='3d'),np.mean([PAR[c] for c in CL],0),'all_five mean')
    plt.savefig('test.png',dpi=90)
