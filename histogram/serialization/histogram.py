import sys
import os
from distutils.util import strtobool

import datetime
import numpy as np
import h5py

from .. import Histogram, HistogramAxis, rc

def ask_overwrite(filepath):
    global rc
    if not os.path.exists(filepath):
        dirpath = os.path.dirname(os.path.abspath(filepath))
        if not os.path.exists(dirpath):
            try:
                os.makedirs(dirpath)
            except:
                print('could not create directory: {}'.format(dirpath))
                return False
        return True
    if rc.overwrite.timestamp is not None:
        if (datetime.datetime.now() - rc.overwrite.timestamp).seconds > rc.overwrite.timeout:
            rc.overwrite.overwrite = 'ask'
            rc.overwrite.timestamp = None
    if rc.overwrite.overwrite == 'never':
        return False
    elif rc.overwrite.overwrite == 'always':
        return True
    else:
        sys.stdout.write('Overwrite file {}? [Yes, No, nEver, Always]\n'.format(filepath))
        while True:
            intext = input().lower()
            try:
                return strtobool(intext)
            except ValueError:
                if intext in ['never','e','none']:
                    rc.overwrite.overwrite = 'never'
                    rc.overwrite.timestamp = datetime.datetime.now()
                    return False
                elif intext in ['always','a','all']:
                    rc.overwrite.overwrite = 'always'
                    rc.overwrite.timestamp = datetime.datetime.now()
                    return True
                sys.stdout.write('Please respond with: Yes, No, nEver, Always.\n')
                sys.stdout.write('nEver and Always will be stickied for {} minutes'.format(int(rc.overwrite.timeout / 60.))+'\n')


def save_histogram_to_hdf5_group(grp, hist, **kwargs):
    grp.create_dataset('data',
        hist.data.shape, hist.data.dtype, data=hist.data[...])
    if hist.uncert is not None:
        grp.create_dataset('uncert',
            hist.uncert.shape, hist.uncert.dtype, data=hist.uncert[...])
    for i,ax in enumerate(hist.axes):
        edge = grp.create_dataset('edges'+str(i),
            (len(ax.edges),), 'f', data=ax.edges)
        if ax.label is not None:
            edge.attrs['label'] = ax.label
    if hist.label is not None:
        grp.attrs['label'] = hist.label
    if hist.title is not None:
        grp.attrs['title'] = hist.title

def load_histogram_from_hdf5_group(grp):
    data = grp['data']
    axes = []
    for i in range(len(data.shape)):
        edges = grp['edges{}'.format(i)]
        axes += [HistogramAxis(edges,label=edges.attrs.get('label', None))]
    return Histogram(
        *axes,
        data=data,
        uncert = grp.get('uncert',None),
        title = grp.attrs.get('title', None),
        label = grp.attrs.get('label', None))


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
        return load_histogram_from_hdf5(filepath)
    else:
        return load_histogram_from_npz(filepath)

Histogram.load = load_histogram


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
