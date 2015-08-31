import ROOT

from .. import Histogram, HistogramAxis

def asroot(hist, name):
    '''Convert this histogram to a CERN/ROOT object (TH1F, TH2F, etc)'''
    if hist.dim > 3:
        raise TypeError('Can not convert histogram with dimensions > 3 to ROOT')

    title = hist.title if hist.title is not None else name

    args = [name,title]
    for ax in hist.axes:
        argx.append(ax.nbins, ax.edges)

    hnew_dispatch = {
        1: ROOT.TH1D,
        2: ROOT.TH2D,
        3: ROOT.TH3D }

    hnew = hnew_dispatch[hist.dim](*args)

    hnew.SetContent(hist.data)
    if hist.uncert is not None:
        hnew.SetError(hist.uncert)

    axlabel_dispatch = {
        0: hnew.GetXaxes().SetTitle,
        1: hnew.GetYaxes().SetTitle,
        2: hnew.GetZaxes().SetTitle }

    for i,ax in enumerate(hist.axes):
        if ax.label is not None:
            axlabel_dispatch[i](ax.label)

    if hist.label is not None:
        if hist.dim < 3:
            axlabel_dispatch[hist.dim](hist.label)
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
    dim = hist.GetDimension()
    for i in range(dim):
        ax = getax_dispatch[i]()
        nbins = ax.GetNbins()
        edges = [ax.GetBinLowEdge(b) for b in range(nbins)]
        edges.append(ax.GetBinHighEdge(nbins-1))
        label = ax.GetTitle()

        axes.append(HistogramAxis(edges, label=label))

    title = hist.GetTitle()
    label = getax_dispatch[dim]().GetTitle()

    return Histogram(*axes,label=label,title=title)

Histogram.fromroot = fromroot
