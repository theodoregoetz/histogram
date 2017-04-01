# coding: utf-8
import logging
import os
import warnings

from .. import Histogram, rc
from .ask_overwrite import ask_overwrite
from .histogram_numpy import save_histogram_to_npz, load_histogram_from_npz

log = logging.getLogger(__name__)

try:
    from .histogram_hdf5 import (
        save_histogram_to_hdf5, load_histogram_from_hdf5,
        save_histograms_to_hdf5, load_histograms_from_hdf5)
    have_h5py = True
except ImportError:
    have_h5py = False
    warnings.warn('Could not import h5py. You will not be able to load/save'
                  ' histograms stored in hdf5 format.', ImportWarning)

try:
    from .histogram_root import save_histogram_to_root, load_histogram_from_root
    have_pyroot = True
except ImportError:
    have_pyroot = False
    warnings.warn('Could not import ROOT (python bindings to CERN\'s ROOT).'
                  ' You will not be able to convert to/from ROOT format.',
                  ImportWarning)


def save_histogram(hist, filepath, prefix=None, **kwargs):
    if prefix is not None:
        filepath = os.path.join(prefix, filepath)
    elif rc.histdir is not None:
        if not os.path.isabs(filepath):
            filepath = os.path.join(rc.histdir, filepath)
    if not any(filepath.endswith(x) for x in ['.hist', '.npz', '.hdf5',
                                              '.h5', '.root']):
        filepath += '.hist'
    if not ask_overwrite(filepath):
        log.warning('not overwriting {}'.format(filepath))
    else:
        if any(filepath.endswith(x) for x in ['.hdf5', '.h5']):
            if not have_h5py:
                raise ImportError('Missing module: h5py')
            save_histogram_to_hdf5(hist, filepath, **kwargs)
        elif filepath.endswith('.root'):
            if not have_pyroot:
                raise ImportError('Missing module: ROOT')
            save_histogram_to_root(hist, filepath, **kwargs)
        else:
            save_histogram_to_npz(hist, filepath, **kwargs)

Histogram.save = save_histogram


@staticmethod
def load_histogram(filepath, prefix=None):
    if prefix is not None:
        filepath = os.path.join(prefix, filepath)
    elif rc.histdir is not None:
        if not os.path.isabs(filepath):
            filepath = os.path.join(rc.histdir, filepath)
    if not any(filepath.endswith(x) for x in ['.hist', '.npz', '.hdf5', '.h5',
                                              '.root']):
        filepath += '.hist'
    if not os.path.exists(filepath):
        raise Exception(filepath + ' not found.')
    if any(filepath.endswith(x) for x in ['.hdf5', '.h5']):
        if not have_h5py:
            raise ImportError('Missing module: h5py')
        return load_histogram_from_hdf5(filepath)
    elif filepath.endswith('.root'):
        if not have_pyroot:
            raise ImportError('Missing module: ROOT')
        return load_histogram_from_root(filepath)
    else:
        return load_histogram_from_npz(filepath)

Histogram.load = load_histogram


def save_histograms(hdict, filepath, prefix=None):
    if prefix is not None:
        filepath = os.path.join(prefix, filepath)
    elif rc.histdir is not None:
        if not os.path.isabs(filepath):
            filepath = os.path.join(rc.histdir, filepath)
    if not any(filepath.endswith(x) for x in ['.hdf5', '.h5']):
        filepath += '.hdf5'
    if not ask_overwrite(filepath):
        log.warning('not overwriting {}'.format(filepath))
    else:
        if not have_h5py:
            raise ImportError('Missing module: h5py')
        save_histograms_to_hdf5(hdict, filepath)


def load_histograms(filepath, prefix=None):
    if prefix is not None:
        filepath = os.path.join(prefix, filepath)
    elif rc.histdir is not None:
        if not os.path.isabs(filepath):
            filepath = os.path.join(rc.histdir, filepath)
    if not any(filepath.endswith(x) for x in ['.hdf5', '.h5']):
        filepath += '.hdf5'
    if not os.path.exists(filepath):
        raise Exception(filepath + ' not found.')
    if not have_h5py:
        raise ImportError('Missing module: h5py')
    return load_histograms_from_hdf5(filepath)
