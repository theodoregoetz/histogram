import sys
import os

import numpy as np

from .. import Histogram, HistogramAxis, rc
from .ask_overwrite import ask_overwrite

try:
    from .histogram_hdf5 import *
    have_h5py = True
except ImportError:
    have_h5py = False
    sys.stderr.write('''\
    Warning: Could not import h5py.
    You will not be able to load/save histograms stored in hdf5 format.
    ''')


def save_histogram_to_npz(filepath, hist, **kwargs):
    '''
    saves a Histogram object to a file
    in npz format
    '''
    fout = open(filepath,'wb')
    np.savez(fout, **hist.asdict(**kwargs))
    fout.close()

def load_histogram_from_npz(filepath):
    '''
    reads in a Histogram object from a file
    in npz format
    '''
    return Histogram.fromdict(**np.load(filepath))


def save_histogram(filepath, hist, prefix=None, **kwargs):
    if prefix is not None:
        filepath = os.path.join(prefix,filepath)
    elif rc.histdir is not None:
        if not os.path.isabs(filepath):
            filepath = os.path.join(rc.histdir,filepath)
    if not any(filepath.endswith(x) for x in ['.hist','.npz','.hdf5','.h5']):
        filepath += '.hist'
    if not ask_overwrite(filepath):
        print('not overwriting {}'.format(filepath))
    else:
        if any(filepath.endswith(x) for x in ['.hdf5','.h5']):
            global have_h5py
            if not have_h5py:
                raise ImportError('Missing module: h5py')
            save_histogram_to_hdf5(filepath,hist,**kwargs)
        else:
            save_histogram_to_npz(filepath,hist,**kwargs)

Histogram.save = lambda self,filepath,prefix=None,**kw: save_histogram(filepath,self,prefix=prefix,**kw)

@staticmethod
def load_histogram(filepath, prefix=None):
    if prefix is not None:
        filepath = os.path.join(prefix,filepath)
    elif rc.histdir is not None:
        if not os.path.isabs(filepath):
            filepath = os.path.join(rc.histdir,filepath)
    if not any(filepath.endswith(x) for x in ['.hist','.npz','.hdf5','.h5']):
        filepath += '.hist'
    if not os.path.exists(filepath):
        raise Exception(filepath+' not found.')
    if any(filepath.endswith(x) for x in ['.hdf5','.h5']):
        global have_h5py
        if not have_h5py:
            raise ImportError('Missing module: h5py')
        return load_histogram_from_hdf5(filepath)
    else:
        return load_histogram_from_npz(filepath)

Histogram.load = load_histogram
