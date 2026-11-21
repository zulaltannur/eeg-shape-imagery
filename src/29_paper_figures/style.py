"""Shared style and helpers for the manuscript figures (stage 29)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')); import participant_order  # original participant order (see src/participant_order.py)
import json, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)
INK = '#1f2328'; MUT = '#6e7781'; GRID = '#d0d7de'
CL = ['cone', 'cube', 'sphere', 'pyramid', 'cylinder']
GROUPS = ['Sighted', 'Congenitally low vision', 'Later vision loss', 'Congenitally blind']
GCOL = {'Sighted': '#2f6fde', 'Congenitally low vision': '#e07b39',
        'Later vision loss': '#2a9d8f', 'Congenitally blind': '#8a63d2'}
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.edgecolor': GRID,
                     'axes.labelcolor': INK, 'xtick.color': MUT, 'ytick.color': MUT,
                     'axes.titlecolor': INK, 'savefig.dpi': 300})
def anon(code, lang='en'):
    """Participant label in the requested language: P01.. (English) or K01.. (Turkish), same number."""
    return ('P' if lang == 'en' else 'K') + code[1:]

def res(*p):
    return os.path.join(ROOT, 'results', *p)

def clean(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
