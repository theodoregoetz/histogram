import platform
import sys
import os

import numpy as np

import h5py

from .. import Histogram, HistogramAxis


def create_dataset(group, name, data):
    return group.create_dataset(name, data.shape, data.dtype, data[...])


def save_histogram_hdf5_group(hist, grp):
    create_dataset(grp, 'data', hist.data)
    if hist.has_uncert:
        create_dataset(grp, 'uncert', hist.uncert)
    for i, ax in enumerate(hist.axes):
        edge = create_dataset(grp, 'edges{}'.format(i), ax.edges)
        if ax.label is not None:
            edge.attrs['label'] = ax.label
    if hist.label is not None:
        grp.attrs['label'] = hist.label
    if hist.title is not None:
        grp.attrs['title'] = hist.title


def load_histogram_hdf5_group(grp):
    data = grp['data']
    axes = []
    for i in range(len(data.shape)):
        edges = grp['edges{}'.format(i)]
        label = edges.attrs.get('label', None)
        axes.append(HistogramAxis(edges, label=label))
    label = grp.attrs.get('label', None)
    title = grp.attrs.get('title', None)
    return Histogram(
        *axes,
        data=data,
        uncert=grp.get('uncert', None),
        label=label,
        title=title)


def save_histogram_hdf5(hist, filepath):
    '''
    saves a Histogram object to a file
    in hdf5 format
    '''
    with h5py.File(filepath, 'w') as h5file:
        save_histogram_hdf5_group(hist, h5file)


def load_histogram_hdf5(filepath):
    '''
    reads in a Histogram object from a file
    in hdf5 format
    '''
    with h5py.File(filepath, 'r') as h5file:
        return load_histogram_hdf5_group(h5file)


def save_histograms_hdf5(hdict, filepath):
    '''
    saves a dict{str_name : Histogram} object to a file
    in hdf5 format
    '''
    with h5py.File(filepath, 'w') as h5file:
        for hname in hdict:
            hist = hdict[hname]
            grp = h5file.create_group(hname)
            save_histogram_hdf5_group(hist, grp)


def load_histograms_hdf5(filepath):
    '''
    reads in a dict{str_name : Histogram} object from a file
    in hdf5 format
    '''
    with h5py.File(filepath, 'r') as h5file:
        h = {}
        for grp in h5file:
            h[grp] = load_histogram_hdf5_group(h5file[grp])
        return h
