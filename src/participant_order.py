"""Participant order used in the original analyses.

Several analyses draw random subsets of participants from sorted lists, so their exact numbers
depend on the order in which participants are sorted. The original analyses sorted participants
by internal codes that are not published. The data package shared on request contains
`participants/analysis_order.json`, the anonymous labels (P01..P23) in that original order.
When the file is present, `sorted()` on a collection of participant labels follows this order,
which reproduces the reported numbers exactly; without it, ordinary sorting is used and
results that depend on random participant subsets can differ slightly.
"""
import builtins, json, os, re

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_FILE = os.path.join(_ROOT, 'participants', 'analysis_order.json')
_ORDER = {c: i for i, c in enumerate(json.load(open(_FILE)))} if os.path.exists(_FILE) else None
_LABEL = re.compile(r'^[PX]\d\d$')
_sorted = getattr(builtins, '_original_sorted', builtins.sorted)


def _participant_sorted(iterable, *args, **kwargs):
    items = list(iterable)
    if (_ORDER and not args and not kwargs.get('key') and items
            and all(isinstance(x, str) and _LABEL.match(x) and x in _ORDER for x in items)):
        return _sorted(items, key=_ORDER.__getitem__, reverse=kwargs.get('reverse', False))
    return _sorted(items, *args, **kwargs)


if _ORDER is not None:
    builtins._original_sorted = _sorted
    builtins.sorted = _participant_sorted
