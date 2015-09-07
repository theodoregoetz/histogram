import io

import numpy as np

from .. import Histogram
from ..detail.strings import encode_dict, decode_dict

def save_histogram_to_npz(filepath, hist):
    '''
    saves a Histogram object to a file
    in npz format
    '''
    fout = io.open(filepath,'wb')
    np.savez(fout, **encode_dict(hist.asdict()))
    fout.close()

def load_histogram_from_npz(filepath):
    '''
    reads in a Histogram object from a file
    in npz format
    '''
    data = dict(np.load(filepath))
    for k in data:
        if data[k].dtype.kind == 'S':
            data[k] = data[k].tostring().decode('utf-8')
        elif data[k].dtype.kind == 'U':
            data[k] = data[k].tostring().decode('utf-32')
    return Histogram.fromdict(**decode_dict(data))
