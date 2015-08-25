import numpy as np

from matplotlib import pyplot
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import \
    zoomed_inset_axes, mark_inset

from .cmap import flame

from .. import Histogram, HistogramAxis

def plothist_grid(hist, xaxis=0, yaxis=1, **kwargs):
    debug = kwargs.pop('debug',False)
    if hist.dim != 3:
        raise Exception('histogram must be of dimension 3')
    if not {xaxis,yaxis}.issubset({0,1,2}):
        raise Exception('x and y axes must be one of 0,1,2')
    if xaxis == yaxis:
        raise Exception('x and y axes must be different')

    hook = kwargs.pop('hook', None)

    subplots_kw = dict(
        figsize = kwargs.pop('figsize',(12,12)),
        sharex = kwargs.pop('sharex',True),
        sharey = kwargs.pop('sharey',False))

    if 'style' not in kwargs:
        kwargs['style'] = 'polygon'

    zaxis = ({0,1,2} - {xaxis,yaxis}).pop()

    if debug:
        print('rolling axes...',flush=True)
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

    if debug:
        print('setting up axes... ({}, {})'.format(ybins,xbins),flush=True)
    fig, ax = pyplot.subplots(ybins, xbins, **subplots_kw)
    # for small font: bottom=.12
    fig.subplots_adjust(bottom=0.15, hspace=0, wspace=0)

    if hook != None:
        hook_output = None
        hook_args = kwargs.pop('hook_args',None)
    if debug:
        print('setting up tick marks...',flush=True)
    for a in range(ybins):
        for b in range(xbins):
            ax[a,b].set_yticks([])
            ax[a,b].set_yticklabels([])
            pyplot.setp(ax[a,b].get_xticklabels(), visible=False)

    if debug:
        print('setting up axtot...',flush=True)
    axtot = fig.add_subplot(1,1,1)
    axtot.axesPatch.set_visible(False)

    axtot.xaxis.set_ticks_position('bottom')
    axtot.yaxis.set_ticks_position('left')
    # for small font: bottom outward: 55
    axtot.spines['bottom'].set_position(('outward',65))
    axtot.spines['left'].set_position(('outward',30))

    axtot.set_xlim(xaxis.limits)
    axtot.set_ylim(yaxis.limits)
    axtot.set_xlabel(xaxis.label)
    axtot.set_ylabel(yaxis.label)
    if hist.title is not None:
        axtot.set_title(hist.title)

    # disable panning and zooming for the overall axes
    axtot.set_xlim = lambda *args,**kwargs: None
    axtot.set_ylim = lambda *args,**kwargs: None

    # do not rasterize. this is all text and lines
    axtot.rasterize = False

    zaxis_nolabel = HistogramAxis(zaxis.edges[:])
    h = Histogram(zaxis_nolabel)

    if debug:
        print('looping over axes',end='...',flush=True)
    for yi in range(ybins):
        for xi in range(xbins):
            # x goes from left to right which is fine,
            # but y goes from top to bottom which is
            # opposite of how we want the grid to be
            # presented so we reverse y for the axis
            # but keep y for the histogram
            a = ybins - yi - 1

            h.data[...] = data[xi,yi,:]
            if uncert is not None:
                h.uncert[...] = uncert[xi,yi,:]

            if hook is not None:
                kw = {}
                if hook_args is not None:
                    kw['hook_args'] = [x[xi][yi] for x in hook_args]

                if debug:
                    print(yi,xi,flush=True)
                hout = hook(ax[a,xi], h, **kw)

                if hook_output == None:
                    hook_output = [[x[:]
                        for x in [[None]*ybins]*xbins]
                            for i in range(len(hout))]

                for i,o in enumerate(hout):
                    hook_output[i][xi][yi] = o

            else:
                ax[a,xi].plothist(h, **kwargs)

                totals = str(int(h.sum()))
                ax[a,xi].text(0.1, 0.9, totals,
                    verticalalignment='top',
                    transform=ax[a,xi].transAxes)

    if debug:
        print('done looping for axes.',flush=True)

    if len(ax[0]) < 5:

        pyplot.setp(ax[ybins-1,0].get_xticklabels(), visible=True)
        ax[ybins-1,0].set_xlabel(zaxis.label)
        axins = None

    else:

        ax1 = ax[ybins-1,round(0.35*len(ax[0]))]

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
        axins.set_xlabel(zaxis.label)

        axins.set_rasterized(False)
        axins.rasterize = False

    if hook == None:
        return fig, ax, axtot, axins
    else:
        return (fig, ax, axtot, axins), hook_output



if __name__ == '__main__':
    from pyhep import Histogram
    from matplotlib import cm

    def hook(ax,h,**kwargs):
        c,a = kwargs.pop('hook_args',None)
        ax.plothist(h, color=cm.jet(c), alpha=a)
        return (h.sum(),h.data.std())


    d0 = (10,[0,100],'x')
    d1 = (9, [0,100],'y')
    d2 = (100,[-0.5,100.5],'z')
    h3 = Histogram(d0,d1,d2,'hist title','hist label')
    h3.data = np.random.uniform(0,100,h3.shape())

    cols = np.random.randint(0,255,(10,9))
    alphas = np.random.uniform(0,1,(10,9))

    figax, hout = plothist_grid(h3, hook=hook, hook_args=(cols,alphas))
    sums,stds = hout

    print(np.array(sums).shape)
    print(np.array(stds).shape)
    pyplot.show()
