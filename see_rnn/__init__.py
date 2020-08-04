import os

def _get_scales():
    s = os.environ.get('SCALEFIG', '1')
    os.environ['SCALEFIG'] = s
    if ',' in s:
        w_scale, h_scale = map(float, s.strip('[()]').split(','))
    else:
        w_scale, h_scale = float(s), float(s)
    return w_scale, h_scale

def scalefig(fig):
    """Used internally to scale figures according to env var 'SCALEFIG'.

    os.environ['SCALEFIG'] can be an int, float, tuple, list, or bracketless
    tuple, but must be a string: '1', '1.1', '(1, 1.1)', '1,1.1'.
    """
    w, h = fig.get_size_inches()
    w_scale, h_scale = _get_scales()  # refresh in case env var changed
    fig.set_size_inches(w * w_scale, h * h_scale)

##############################################################################

from . import visuals_gen
from . import visuals_rnn
from . import inspect_gen
from . import inspect_rnn

from .visuals_gen import *
from .visuals_rnn import *
from .inspect_gen import *
from .inspect_rnn import *

__version__ = '1.15.0'
