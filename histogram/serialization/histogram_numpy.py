import io

import numpy as np

from .. import Histogram

def save_histogram_npz(hist, filepath):
    '''
    saves a Histogram object to a file
    in npz format
    '''
    with io.open(filepath, 'wb') as fout:
        np.savez(fout, **hist.asdict('utf-8', flat=True))

def load_histogram_npz(filepath):
    '''
    reads in a Histogram object from a file
    in npz format
    '''
    hdict = dict(np.load(filepath, encoding='bytes'))
    for k, v in hdict.items():
        if v.dtype.char in ['S', 'U']:
            hdict[k] = v.tostring()
    return Histogram.fromdict(hdict, 'utf-8')
