import io

import numpy as np

from .. import Histogram

def save_histogram_to_npz(filepath, hist):
    '''
    saves a Histogram object to a file
    in npz format
    '''
    with io.open(filepath,'wb') as fout:
        np.savez(fout, **hist.asdict())

def load_histogram_from_npz(filepath):
    '''
    reads in a Histogram object from a file
    in npz format
    '''
    return Histogram.fromdict(dict(np.load(filepath)))
