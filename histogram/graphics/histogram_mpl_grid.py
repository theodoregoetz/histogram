import math
from copy import copy
import numpy as np

import matplotlib
from matplotlib.gridspec import GridSpec


import numpy as np

import matplotlib
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import \
    zoomed_inset_axes, mark_inset

from .. import Histogram, HistogramAxis
from .detail import mpl_subplots

def plothist_grid(fig, hist, xaxis=0, yaxis=1, hook=None, hook_kw={}, **kwargs):
    assert hist.dim == 3, 'histogram must be of dimension 3'

    if 'style' not in kwargs:
        kwargs['style'] = 'polygon'

    zaxis = ({0,1,2} - {xaxis,yaxis}).pop()

    if (xaxis,yaxis) == (0,1):
        data = hist.data
        uncert = hist.uncert
    else:
        a = yaxis
        if xaxis < yaxis:
            b = xaxis + 1
        else:
            b = xaxis
        data = np.rollaxis(np.rollaxis(hist.data,a),b)
        if hist.uncert != None:
            uncert = np.rollaxis(np.rollaxis(hist.uncert,a),b)
        else:
            uncert = None

    xaxis = hist.axes[xaxis]
    yaxis = hist.axes[yaxis]
    zaxis = hist.axes[zaxis]

    xbins = xaxis.nbins
    ybins = yaxis.nbins

    axs = fig.subplots(ybins, xbins,
        sharex = kwargs.pop('sharex','all'),
        sharey = kwargs.pop('sharey','none'),
        subplot_kw = kwargs.pop('subplot_kw',None))

    if hook is not None:
        hook_output = [[None for i in ybins] for j in xbins]

    for ax in axs.flat:
        ax.set_autoscale_on(True)
        for tl in ax.get_xticklabels():
            tl.set_visible(False)
        for tl in ax.get_yticklabels():
            tl.set_visible(False)

    axtot = fig.add_subplot(1,1,1)
    axtot.axesPatch.set_visible(False)

    axtot.xaxis.set_ticks_position('bottom')
    axtot.yaxis.set_ticks_position('left')
    # for small font: bottom outward: 55
    axtot.spines['bottom'].set_position(('outward',65))
    axtot.spines['left'].set_position(('outward',20))

    axtot.set_xlim(xaxis.limits)
    axtot.set_ylim(yaxis.limits)

    # disable panning and zooming for the overall axes
    axtot.set_xlim = lambda *args,**kwargs: None
    axtot.set_ylim = lambda *args,**kwargs: None

    zaxis_nolabel = HistogramAxis(zaxis.edges[:])
    h = Histogram(zaxis_nolabel)

    if 'color' in kwargs:
        c = kwargs.pop('color')
        def color(s):
            return c
    else:
        hz = hist.projection(2)
        hzmin = hz.min()
        hzmax = hz.max()
        dz = 0.3 * (hzmax - hzmin)

        norm = Normalize(vmin=hzmin-dz, vmax=hzmax+dz)
        cmap = kwargs.pop('cmap',get_cmap())
        def color(s):
            return cmap(norm(s))

    for yi in range(ybins):
        for xi in range(xbins):
            # x goes from left to right which is fine,
            # but y goes from top to bottom which is
            # opposite of how we want the grid to be
            # presented so we reverse y for the axis
            # but keep y for the histogram
            a = ybins - yi - 1

            h.data[...] = data[xi,yi,...]
            if uncert is not None:
                h.uncert[...] = uncert[xi,yi,...]

            if hook is None:
                axs[a,xi].plothist(h, color=color(h.sum()), **kwargs)
                total = str(int(h.sum()))
                axs[a,xi].text(0.1, 0.9, total,
                    verticalalignment='top',
                    transform=axs[a,xi].transAxes)
            else:
                kw = hook_kw[xi][yi]
                kw.update(kwargs)
                hook_output[xi][yi] = hook(axs[a,xi], h, **kw)

    if xbins < 5:
        for xtl in axs[ybins-1,0].get_xticklabels():
            xtl.set_visible(True)

        if zaxis.label is not None:
            axs[ybins-1,0].set_xlabel(zaxis.label)

        axins = None

    else:

        ax1 = axs[ybins-1,round(0.35*xbins)]

        # zoom factor is such that about half the space is used
        zoom_factor = float(xbins) / 2.

        # create the zoomed in axis based on the original axes
        axins = zoomed_inset_axes(parent_axes=ax1, zoom=zoom_factor, loc=8,
                                  bbox_to_anchor=(0.5, 0.),
                                  bbox_transform=ax1.transAxes,
                                  axes_kwargs=dict(sharex=ax1, sharey=ax1),
                                  borderpad=-1.5, # in fraction of font size
                                  )

        # dotted lines to show that this is the inset-axis
        pp, p1, p2 = mark_inset(parent_axes=ax1, inset_axes=axins,
                                loc1=3, loc2=4.,
                                linestyle="dotted")

        # only want to draw some of the lines
        pp.set_visible(False)

        # hide the plotting area
        axins.axesPatch.set_zorder(-100)
        axins.axesPatch.set_visible(False)
        axins.axesPatch.set_color('white')
        axins.patch.set_facecolor('white')

        # we want to draw the bottom spine only
        axins.set_frame_on(True)
        axins.spines['top'].set_visible(False)
        axins.spines['left'].set_visible(False)
        axins.spines['right'].set_visible(False)

        # don't draw the y axis ticks or labels
        axins.set_yticks([])
        axins.set_yticklabels([])

        # only draw the bottom (x) axis
        axins.xaxis.set_ticks_position('bottom')
        axins.xaxis.set_label_position('bottom')

        # The inset axis label
        if zaxis.label is not None:
            axins.set_xlabel(zaxis.label)

    if xaxis.label is not None:
        axtot.set_xlabel(xaxis.label)
    if yaxis.label is not None:
        axtot.set_ylabel(yaxis.label)
    if hist.title is not None:
        axtot.set_title(hist.title)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0,wspace=0)

    if hook is not None:
        return (ax, axtot, axins), hook_output
    else:
        return ax, axtot, axins

matplotlib.figure.Figure.plothist_grid = plothist_grid
