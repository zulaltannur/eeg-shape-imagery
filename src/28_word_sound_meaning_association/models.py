import os as _os
if not globals().get('_RUN_DIR_SET'): _os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'results', '28_word_sound_meaning_association'))  # run from this stage's results folder (skipped when this file is executed inside another stage's script)
_RUN_DIR_SET = True
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
# Model dissimilarity matrices for the words (10 shape pairs)
import numpy as np,itertools,json
CL=['cone','cube','sphere','pyramid','cylinder'];P=list(itertools.combinations(range(5),2))
def rdm_from(X,metric='euc'):
    X=np.array(X,float)
    if metric=='cos': X=X/np.linalg.norm(X,axis=1,keepdims=True);return np.array([1-X[i]@X[j] for i,j in P])
    X=(X-X.mean(0))/(X.std(0)+1e-9);return np.array([np.linalg.norm(X[i]-X[j]) for i,j in P])
M={}
# SOUND (phonological; from the written form because the sound files were not available): number of phonemes, syllables, first consonant, first vowel
n_phonemes=[4,3,4,7,8];syllable=[2,1,2,3,3];first=['k','k','k','p','s'];vowel=['o','y','y','i','i']   # first vowel, IPA
M['sound · length (number of phonemes)']=np.array([abs(n_phonemes[i]-n_phonemes[j]) for i,j in P],float)
M['sound · first sound and vowel']=np.array([(first[i]!=first[j])+(vowel[i]!=vowel[j]) for i,j in P],float)
# MEANING
g=np.load('../../results/08_coordinate_geometry/geom.npz',allow_pickle=True);gc=list(g['CL']);ix=[gc.index(c) for c in CL]
M['meaning · geometry (corner, edge, face, curvature, pointedness)']=rdm_from(g['desc_raw'][ix][:,:5])
M['meaning · language model (concept similarity)']=rdm_from(np.load('../../data/derived/07_things/multilingual_text.npz')['en'],'cos')
M['meaning · curved/cornered']=np.array([float([1,0,1,0,1][i]!=[1,0,1,0,1][j]) for i,j in P])
M['meaning · pointed/not']=np.array([float([1,0,0,1,0][i]!=[1,0,0,1,0][j]) for i,j in P])
# ASSOCIATION (coded from the experiment notes; category counts: food, games-toys-sport, building-monument, nature-world, container-kitchen, accessory-ornament, traffic-vehicle, stationery-technology)
C={'cone':[4,1,0,0,2,6,2,0],'cube':[2,7,2,0,2,0,0,1],'sphere':[1,7,0,7,0,3,0,0],'pyramid':[0,4,16,1,0,0,0,0],'cylinder':[0,1,2,2,6,0,2,3]}
M['association (everyday objects in the notes)']=rdm_from([C[c] for c in CL],'cos')
json.dump({k:v.tolist() for k,v in M.items()},open('models.json','w'),ensure_ascii=False,indent=1)
from scipy.stats import spearmanr
ks=list(M);print('relations between models');[print(f'{a[:28]:28s} {b[:28]:28s} {spearmanr(M[a],M[b]).correlation:+.2f}') for a,b in itertools.combinations(ks,2)]
