# coding: utf-8
import platform
import sys
import os

import numpy as np

import h5py

from .. import Histogram, HistogramAxis, rc
from .ask_overwrite import ask_overwrite


def create_dataset(group, name, data):
    return group.create_dataset(name, data.shape, data.dtype, data[...])


def save_histogram_to_hdf5_group(grp, hist):
    create_dataset(grp, 'data', hist.data)
    if hist.has_uncert:
        create_dataset(grp, 'uncert', hist.uncert)
    for i, ax in enumerate(hist.axes):
        edge = create_dataset(grp, 'edges{}'.format(i), ax.edges)
        if ax.label is not None:
            edge.attrs['label'] = ax.label.encode()
    if hist.label is not None:
        grp.attrs['label'] = hist.label.encode()
    if hist.title is not None:
        grp.attrs['title'] = hist.title.encode()


def load_histogram_from_hdf5_group(grp):
    data = grp['data']
    axes = []
    for i in range(len(data.shape)):
        edges = grp['edges{}'.format(i)]
        label = edges.attrs.get('label', None)
        if label is not None:
            label = label.decode()
        axes.append(HistogramAxis(edges, label=label))
    label = grp.attrs.get('label', None)
    if label is not None:
        label = label.decode()
    title = grp.attrs.get('title', None)
    if title is not None:
        title = title.decode()
    return Histogram(
        *axes,
        data=data,
        uncert=grp.get('uncert', None),
        label=label,
        title=title)


def save_histogram_to_hdf5(filepath, hist):
    '''
    saves a Histogram object to a file
    in hdf5 format
    '''
    with h5py.File(filepath, 'w') as h5file:
        save_histogram_to_hdf5_group(h5file, hist)


def load_histogram_from_hdf5(filepath):
    '''
    reads in a Histogram object from a file
    in hdf5 format
    '''
    with h5py.File(filepath, 'r') as h5file:
        return load_histogram_from_hdf5_group(h5file)


def save_histograms(filepath, prefix=None, **hdict):
    '''
    saves a dict{str_name : Histogram} object to a file
    in hdf5 format
    '''
    if prefix is not None:
        filepath = os.path.join(prefix, filepath)
    elif rc.histdir is not None:
        if os.path.isabs(filepath):
            filepath = os.path.join(rc.histdir, filepath)
    if not ask_overwrite(filepath):
        print('not overwriting {}'.format(filepath))
    else:
        if os.path.exists(filepath):
            os.remove(filepath)
        with h5py.File(filepath, 'w') as h5file:
            for hname in hdict:
                hist = hdict[hname]
                grp = h5file.create_group(hname)
                save_histogram_to_hdf5_group(grp, hist)


def load_histograms(filepath, prefix=None):
    '''
    reads in a dict{str_name : Histogram} object from a file
    in hdf5 format
    '''
    if prefix is not None:
        filepath = os.path.join(prefix, filepath)
    elif rc.histdir is not None:
        if os.path.isabs(filepath):
            filepath = os.path.join(rc.histdir, filepath)
    if not os.path.exists(filepath):
        raise Exception(filepath+' not found.')
    with h5py.File(filepath, 'r') as h5file:
        h = {}
        for grp in h5file:
            h[grp] = load_histogram_from_hdf5_group(h5file[grp])
        return h
