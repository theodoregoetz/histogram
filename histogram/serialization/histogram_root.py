from warnings import warn
import numpy as np

import ROOT

from .. import Histogram, HistogramAxis
from ..detail.strings import encoded_str, decoded_str

def asroot(hist, name):
    '''Convert this histogram to a CERN/ROOT object (TH1F, TH2F, etc)'''
    if hist.dim > 3:
        raise TypeError('Can not convert histogram with dimensions > 3 to ROOT')

    title = encoded_str(hist.title) if hist.title is not None else encoded_str(name)

    args = [name,title]
    for ax in hist.axes:
        args.append(ax.nbins)
        args.append(ax.edges)

    hnew_dispatch = {
        1: ROOT.TH1D,
        2: ROOT.TH2D,
        3: ROOT.TH3D }

    hnew = hnew_dispatch[hist.dim](*args)

    hnew.SetContent(hist.data.astype(np.float64))
    if hist.uncert is not None:
        hnew.SetError(hist.uncert.astype(np.float64))

    axlabel_dispatch = {
        0: hnew.GetXaxis().SetTitle,
        1: hnew.GetYaxis().SetTitle,
        2: hnew.GetZaxis().SetTitle }

    for i,ax in enumerate(hist.axes):
        if ax.label is not None:
            axlabel_dispatch[i](encoded_str(ax.label))

    if hist.label is not None:
        if hist.dim < 3:
            axlabel_dispatch[hist.dim](encoded_str(hist.label))
        else:
            warn('CERN/ROOT 3D Histograms do not store a content label. Information (hist.label = \''+hist.label+'\') has been lost.')

    return hnew

Histogram.asroot = asroot

@staticmethod
def fromroot(hist):
    assert isinstance(hist, ROOT.TH1), 'Object must inheret from TH1'

    getax_dispatch = {
        0: hist.GetXaxis,
        1: hist.GetYaxis,
        2: hist.GetZaxis }

    axes = []
    shape = []
    dim = hist.GetDimension()
    for i in range(dim):
        ax = getax_dispatch[i]()
        nbins = ax.GetNbins()
        shape.append(nbins)
        edges = [ax.GetBinLowEdge(b) for b in range(1,nbins+2)]
        label = decoded_str(ax.GetTitle())

        axes.append(HistogramAxis(edges, label=label))

    nbins = np.prod(shape)
    data = np.empty((nbins,), dtype=np.float64)
    for i in range(nbins):
        data[i] = hist.At(i)
    data.shape = shape

    uncert = np.empty((nbins,), dtype=np.float64)
    for i in range(nbins):
        uncert[i] = hist.GetBinError(i)
    uncert.shape = shape

    title = decoded_str(hist.GetTitle())
    label = decoded_str(getax_dispatch[dim]().GetTitle())

    return Histogram(
        *axes,
        label=label,
        title=title,
        data=data,
        uncert=uncert )

Histogram.fromroot = fromroot


def save_histogram_to_root(filepath,hist,mode='RECREATE'):
    fout = ROOT.TFile(filepath,mode)
    h = hist.asroot('h')
    fout.Write('h')
    fout.Close()

def load_histogram_from_root(filepath):
    fin = ROOT.TFile(filepath)
    h = fin.Get('h')
    return Histogram.fromroot(h)
