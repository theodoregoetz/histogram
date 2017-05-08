import math
from copy import copy
import numpy as np

import matplotlib
from matplotlib.gridspec import GridSpec

from .detail import mpl_subplots

def calc_2d_grid_dimensions(ncells):
    a = math.sqrt(ncells)
    if a > 4:
        cols = 5
    elif (a - int(a)) > 0.00001:
        cols = int(a) + 1
    else:
        cols = int(a)

    b = float(ncells) / cols
    if (b - int(b)) > 0.00001:
        rows = int(b) + 1
    else:
        rows = int(b)
    return rows, cols

def plothist_strip(fig, hlist, haxis, hook=None, hook_kw={}, **kwargs):
    assert haxis.isuniform()

    n = len(hlist)
    rows,cols = calc_2d_grid_dimensions(n)

    axs = mpl_subplots.subplots(fig,rows,cols,
        sharex=kwargs.pop('sharex',True),
        sharey=kwargs.pop('sharex','row'),
        subplot_kw = kwargs.pop('subplot_kw',None))
    fig.subplots_adjust(wspace=0, hspace=0.35)

    fig.suptitle(hlist[0].title)

    for h in hlist:
        h.title = None

    if hook is not None:
        hook_output = []

    strip_axs = []
    pts = []
    for r in range(rows):
        ymax = 0
        for c in range(cols):
            i = c + r*cols

            if i < n:

                if hook is None:
                    pts.append(
                        axs[r,c].plothist(
                            hlist[i],
                            style=kwargs.pop('style','errorbar'),
                            **kwargs ) )
                else:
                    kw = hook_kw[i]
                    kw.update(kwargs)
                    hook_output.append(hook(axs[r,c], hlist[i], **kw))

                this_ymax = 1.05 * np.nanmax(hlist[i].data + hlist[i].uncert)
                if this_ymax > ymax:
                    ymax = this_ymax
                    axs[r,0].set_ylim(None,ymax)

        ilow  = r*cols
        ihigh = (r+1)*cols
        if ihigh > n:
            nplots_in_last_row = n % cols
            gs = GridSpec(rows,cols)
            ax = fig.add_subplot(gs[rows-1,:nplots_in_last_row])
        else:
            ax = fig.add_subplot(rows,1,r+1)

        ax.axesPatch.set_alpha(0.)
        ax.yaxis.set_visible(False)

        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        ax.spines['top'].set_position(('outward',7))

        if ilow < len(haxis.edges):
            if ihigh < len(haxis.edges):
                ax.set_xlim(haxis.edges[ilow],haxis.edges[ihigh])
            else:
                ax.set_xlim(haxis.edges[ilow],haxis.edges[-1])
            ax.xaxis.set_ticks(haxis.edges[ilow:ihigh+1])

        # disable interactive pan/zoom for this axis
        ax.set_xlim = lambda *args,**kwargs: None

        strip_axs.append(ax)

    for ax in axs.ravel():
        ax.set_xlabel('')
    for ax in axs.ravel()[-cols:]:
        ax.set_xlabel(hlist[0].axes[0].label)
    for ax in axs[...,0]:
        ax.set_ylabel(hlist[0].label)
    strip_axs[0].set_xlabel(haxis.label, x=0.15)

    for ax in axs.flat:
        ax.label_outer()

    if n < (rows*cols):
        for ax in axs.flat[n:]:
            ax.set_visible(False)
        for ax in axs.flat[n-cols:n]:
            for xt in ax.get_xticklabels():
                xt.set_visible(True)
        o = n % cols
        for ax in axs.flat[n-cols+1:n-cols+o]:
            for lab in ax.get_xticklabels()[:2]:
                lab.set_visible(False)
        axs.flat[n-cols].set_xlabel(hlist[0].axes[0].label)

    for ax in axs[-1,1:]:
        for lab in ax.get_xticklabels()[:2]:
            lab.set_visible(False)

    if hook is not None:
        return (axs, strip_axs), hook_output
    else:
        return axs, strip_axs

matplotlib.figure.Figure.plothist_strip = plothist_strip
