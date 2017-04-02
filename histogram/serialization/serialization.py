# coding: utf-8
import logging
import os
import warnings

from .. import Histogram, rc
from .histogram_numpy import save_histogram_npz, load_histogram_npz


try:
    from .histogram_hdf5 import (
        save_histogram_hdf5, load_histogram_hdf5,
        save_histograms_hdf5, load_histograms_hdf5)
    Histogram.save_hdf5 = save_histogram_hdf5
    Histogram.load_hdf5 = load_histogram_hdf5
    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False
    warnings.warn('Could not import h5py. You will not be able to load/save'
                  ' histograms stored in hdf5 format.', ImportWarning)

try:
    from .histogram_root import save_histogram_root, load_histogram_root
    Histogram.save_root = save_histogram_root
    Histogram.load_root = load_histogram_root
    HAVE_PYROOT = True
except ImportError:
    HAVE_PYROOT = False
    warnings.warn('Could not import ROOT (python bindings to CERN\'s ROOT).'
                  ' You will not be able to convert to/from ROOT format.',
                  ImportWarning)


def save_histogram(hist, filepath):
    if any(filepath.endswith(x) for x in ['.hdf5', '.h5']):
        if not HAVE_H5PY:
            raise ImportError('Missing module: h5py')
        save_histogram_hdf5(hist, filepath)
    elif filepath.endswith('.root'):
        if not HAVE_PYROOT:
            raise ImportError('Missing module: ROOT')
        save_histogram_root(hist, filepath)
    else:
        save_histogram_npz(hist, filepath)


Histogram.save = save_histogram
Histogram.save_npz = save_histogram_npz


@staticmethod
def load_histogram(filepath):
    if any(filepath.endswith(x) for x in ['.hdf5', '.h5']):
        if not HAVE_H5PY:
            raise ImportError('Missing module: h5py')
        return load_histogram_hdf5(filepath)
    elif filepath.endswith('.root'):
        if not HAVE_PYROOT:
            raise ImportError('Missing module: ROOT')
        return load_histogram_root(filepath)
    else:
        return load_histogram_npz(filepath)

Histogram.load = load_histogram
Histogram.load_npz = load_histogram_npz


def save_histograms(hdict, filepath):
    if not HAVE_H5PY:
        raise ImportError('Missing module: h5py')
    save_histograms_hdf5(hdict, filepath)


def load_histograms(filepath):
    if not HAVE_H5PY:
        raise ImportError('Missing module: h5py')
    return load_histograms_hdf5(filepath)
