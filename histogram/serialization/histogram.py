import sys
import os
import io
from warnings import warn

import numpy as np

from .. import Histogram, HistogramAxis, rc
from .ask_overwrite import ask_overwrite

try:
    from .histogram_hdf5 import *
    have_h5py = True
except ImportError:
    have_h5py = False
    warn('Could not import h5py. You will not be able to load/save histograms stored in hdf5 format.', ImportWarning)

try:
    from .histogram_root import *
    have_pyroot = True
except ImportError:
    have_pyroot = False
    warn('Could not import ROOT (python bindings to CERN\'s ROOT). You will not be able to convert to/from PyROOT format.', ImportWarning)


def save_histogram_to_npz(filepath, hist, **kwargs):
    '''
    saves a Histogram object to a file
    in npz format
    '''
    fout = io.open(filepath,'wb')
    np.savez(fout, **hist.asdict(**kwargs))
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
    return Histogram.fromdict(**data)


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
