"""Component spaces of the multi-component model (all from the author's 60 mm CAD geometry)."""
import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '10_multi_component_model'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import numpy as np, json, sys
sys.path.insert(0,'../../src/09_superquadric_bridge'); from sq import surface, PAR
CL=['cone','cube','sphere','pyramid','cylinder']; R=30.0; H=60.0
g=np.load('../../results/08_coordinate_geometry/geom.npz')
# --- VISION: silhouette features from different viewpoints (projection)
from scipy.spatial import ConvexHull
def views():
    out=[]
    for hand in [0,20,45,70]:
        for low in np.arange(0,180,30): out.append((np.radians(hand),np.radians(low)))
    return out
def sil_feats(P):
    F=[]
    for hand,low in views():
        Rz=np.array([[np.cos(low),-np.sin(low),0],[np.sin(low),np.cos(low),0],[0,0,1]]);Rx=np.array([[1,0,0],[0,np.cos(hand),-np.sin(hand)],[0,np.sin(hand),np.cos(hand)]])
        Q=(P@Rz.T@Rx.T)[:,[0,2]];h=ConvexHull(Q);V=Q[h.vertices]
        area=h.volume;per=h.area;comp=4*np.pi*area/per**2
        # contour cornerness: number of points with a large change of angle between neighbouring edges
        d=np.diff(np.vstack([V,V[:1]]),axis=0);ang=np.arctan2(d[:,1],d[:,0]);turn=np.abs((np.diff(np.r_[ang,ang[0]])+np.pi)%(2*np.pi)-np.pi)
        corners=(turn>np.radians(25)).sum()
        w=Q[:,0].max()-Q[:,0].min();hgt=Q[:,1].max()-Q[:,1].min()
        top=Q[Q[:,1]>Q[:,1].max()-0.1*hgt,0];topw=(top.max()-top.min())/w if len(top) else 0
        F.append([area/3600,per/240,comp,corners,hgt/w,topw])
    return np.array(F).mean(0)
VIS=np.array([sil_feats(g[f'pc_{c}']) for c in CL]);vis_names=['silhouette_area','perimeter','roundness','contour_corner','height/width','peak_width']
# --- TOUCH: tactile trace according to exploration procedures
lat=np.sqrt(R**2+R**2+H**2)  # pyramid side edge
HAP=np.array([ # edge length(mm), pointed corner, flat face, curved surface share, largest span(mm), max curvature(1/mm, edge/tip=0.5), grasp: [spherical, wrap, precision tip]
 [2*np.pi*R, 1, 1, 0.77, np.hypot(2*R,H), 0.5, 0,1,1],          # cone
 [12*2*R,    8, 6, 0.0,  np.sqrt(3)*2*R,  0.5, 0,0,0.5],        # cube (power/precision mixed)
 [0,         0, 0, 1.0,  2*R,             1/R, 1,0,0],          # sphere
 [4*2*R+4*lat,5,5, 0.0,  np.hypot(2*R,2*R),0.5, 0,0,1],         # pyramid
 [2*2*np.pi*R,0,2, 0.67, np.hypot(2*R,H), 0.5, 0,1,0]])         # cylinder
hap_names=['edge_length','pointed_corner','flat_face','curved_surface_share','max_span','max_curvature','grasp_spherical','grasp_wrap','grasp_tip']
# --- GEOMETRY: superquadric + structural features
GEO=np.hstack([np.array([PAR[c] for c in CL]),g['desc_z']])
# --- WORD (control): word onset difference in the sound files and number of syllables
WRD=np.array([[1.56,2],[1.56,1],[1.54,2],[2.56,3],[2.73,3]])  # cone, cube, sphere, pyramid, cylinder
def zc(M): return (M-M.mean(0))/(M.std(0)+1e-9)
def rdm(M): Z=zc(M); return np.sqrt(((Z[:,None]-Z[None])**2).sum(-1))
RD={'vision':rdm(VIS),'touch':rdm(HAP),'geometry':rdm(GEO),'word':rdm(WRD),'form(coordinate)':g['chamfer']}
iu=np.triu_indices(5,1);from scipy.stats import spearmanr
np.set_printoptions(precision=2,suppress=True)
print('Vision features\n',dict(zip(CL,np.round(VIS,2).tolist())))
ks=list(RD);C=np.array([[spearmanr(RD[a][iu],RD[b][iu]).correlation for b in ks] for a in ks]);print('Similarity between components (Spearman)\n',ks,'\n',C)
np.savez('comps.npz',CL=CL,**{f'rdm_{k}':v for k,v in RD.items()},VIS=VIS,HAP=HAP,GEO=GEO,WRD=WRD)
