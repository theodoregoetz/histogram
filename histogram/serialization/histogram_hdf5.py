import platform
import sys
import os

import numpy as np

import h5py

from .. import Histogram, HistogramAxis, rc
from .ask_overwrite import ask_overwrite
from .strings import encode_str, decode_str

def save_histogram_to_hdf5_group(grp, hist, **kwargs):
    grp.create_dataset('data',
        hist.data.shape, hist.data.dtype, data=hist.data[...])
    if hist.uncert is not None:
        grp.create_dataset('uncert',
            hist.uncert.shape, hist.uncert.dtype, data=hist.uncert[...])
    for i,ax in enumerate(hist.axes):
        edge = grp.create_dataset('edges{}'.format(i),
            (len(ax.edges),), 'f', data=ax.edges)
        if ax.label is not None:
            edge.attrs['label'] = encode_str(ax.label)
    if hist.label is not None:
        grp.attrs['label'] = encode_str(hist.label)
    if hist.title is not None:
        grp.attrs['title'] = encode_str(hist.title)

def load_histogram_from_hdf5_group(grp):
    data = grp['data']
    axes = []
    for i in range(len(data.shape)):
        edges = grp['edges{}'.format(i)]
        axes.append(
            HistogramAxis(
                np.asarray(edges),
                label = decode_str(edges.attrs.get('label',None)) ) )
    return Histogram(
        *axes,
        data = data,
        uncert = grp.get('uncert',None),
        title = decode_str(grp.attrs.get('title',None)),
        label = decode_str(grp.attrs.get('label',None)) )


def save_histogram_to_hdf5(filepath, hist, **kwargs):
    '''
    saves a Histogram object to a file
    in hdf5 format
    '''
    h5file = h5py.File(filepath, 'w')
    save_histogram_to_hdf5_group(h5file, hist, **kwargs)
    h5file.close()

def load_histogram_from_hdf5(filepath):
    '''
    reads in a Histogram object from a file
    in hdf5 format
    '''
    h5file = h5py.File(filepath, 'r')
    hist = load_histogram_from_hdf5_group(h5file)
    h5file.close()
    return hist

def save_histograms(filepath, prefix=None, **hdict):
    '''
    saves a dict{str_name : Histogram} object to a file
    in hdf5 format
    '''
    if prefix is not None:
        filepath = os.path.join(prefix,filepath)
    elif rc.histdir is not None:
        if os.path.isabs(filepath):
            filepath = os.path.join(rc.histdir,filepath)
    if not ask_overwrite(filepath):
        print('not overwriting {}'.format(filepath))
    else:
        if os.path.exists(filepath):
            os.remove(filepath)
        h5file = h5py.File(filepath, 'w')
        for hname in hdict:
            hist = hdict[hname]
            grp = h5file.create_group(hname)
            save_histogram_to_hdf5_group(grp,hist)
        h5file.close()

def load_histograms(filepath, prefix=None):
    '''
    reads in a dict{str_name : Histogram} object from a file
    in hdf5 format
    '''
    if prefix is not None:
        filepath = os.path.join(prefix,filepath)
    elif rc.histdir is not None:
        if os.path.isabs(filepath):
            filepath = os.path.join(rc.histdir,filepath)
    if not os.path.exists(filepath):
        raise Exception(filepath+' not found.')
    h5file = h5py.File(filepath, 'r')
    h = {}
    for grp in h5file:
        h[grp] = load_histogram_from_hdf5_group(h5file[grp])
    return h
