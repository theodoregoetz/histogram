import math
import numpy as np

from matplotlib import pyplot
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec

from .cmap import flame

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
    return cols, rows

def calc_figure_size(cols,rows):
    return min(cols*5+1,12), min(rows*3+1,8)

def plothist_strip(hist, stripaxis=0, **kwargs):
    if hist.dim not in [2,3]:
        raise Exception('histogram must be of dimension 2 or 3')

    mask = kwargs.pop('mask',None)

    hook = kwargs.pop('hook', None)
    if hook is not None:
        hook_output = None
        hook_args = kwargs.pop('hook_args',None)

    first_index = 0
    for i,d in enumerate(hist.slices_data(stripaxis)):
        nans = ~np.isfinite(d)
        if np.all(nans):
            first_index = i+1
        elif np.allclose(d[~nans],d[~nans].mean()):
            first_index = i+1
        else:
            break

    if first_index > 0:
        stripmin = hist.axes[stripaxis].edges[first_index]
        hist = hist.cut(stripmin,axis=stripaxis)

    saxis = hist.axes[stripaxis]
    sbins = saxis.nbins

    cols,rows = calc_2d_grid_dimensions(sbins)

    fig_kw = dict(
        figsize = kwargs.pop('figsize',calc_figure_size(cols,rows)),
        dpi = kwargs.pop('dpi',90) )

    fig = pyplot.figure(**fig_kw)
    fig.subplots_adjust(left=.07,bottom=.07,right=.95,top=.9,
                        wspace=0,hspace=1.)

    # create all subplots
    # share x with the first plot
    # share y with the first of the current row
    axs = []
    for i in range(sbins):
        kw = {}
        if i > 0:
            kw['sharex'] = axs[0]
            if (i%cols) > 0:
                kw['sharey'] = axs[int(i/cols)*cols]

        axs += [fig.add_subplot(rows,cols,i+1,**kw)]
        axs[-1].locator_params(tight=True, nbins=5)
        axs[-1].xaxis.labelpad = -1

    fig.suptitle(hist.title)
    hist.title = None

    strip_axs = []
    hslices = list(hist.slices(stripaxis))
    if mask is not None:
        masks = [m for m in np.rollaxis(mask,stripaxis)]

    for r in range(rows):
        ymax = 0
        for c in range(cols):
            i = c + r*cols
            if i < len(hslices):

                if hook is None:

                    axs[i].grid(True)
                    if mask is not None:
                        axs[i].plothist(hslices[i], mask=masks[i], style='errorbar')
                    else:
                        axs[i].plothist(hslices[i], style='errorbar')

                else:

                    hook_kw = {}
                    if mask is not None:
                        hook_kw['mask'] = masks[i]

                    if hook_args is not None:
                        hook_kw['hook_args'] = [x[i] for x in hook_args]

                    hout = hook(axs[i], hslices[i], **hook_kw)

                    # setup hook output if it doesn't exist
                    if hook_output is None:
                        hook_output = []
                        for _i in range(len(hout)):
                            tmp = []
                            for _j in range(len(axs)):
                                tmp.append(None)
                            hook_output.append(tmp)

                    for a,o in enumerate(hout):
                        hook_output[a][i] = o

                this_ymax = 1.05 * np.nanmax(hslices[i].data + hslices[i].uncert)
                if this_ymax > ymax:
                    ymax = this_ymax
                    axs[r*cols].set_ylim(None,ymax)

                # hide certain labels
                if c > 0:
                    axs[i].set_xlabel(r'')
                    axs[i].set_ylabel(r'')
                    pyplot.setp(axs[i].get_yticklabels(), visible=False)

                keep_tick = 'first'
                if keep_tick == 'first':
                    # remove the first tick label of all but the
                    # first axis in this row
                    if c > 0:
                        xticks = axs[i].xaxis.get_major_ticks()
                        for t in xticks[:1]:
                            t.label1.set_visible(False)
                else: # if keep_tick == 'last':
                    # remove the last tick label of all but the
                    # last axis in this row
                    if (c+1) < cols:
                        xticks = axs[i].xaxis.get_major_ticks()
                        for t in xticks[-2:]:
                            t.label1.set_visible(False)

        ilow  = r*cols
        ihigh = (r+1)*cols
        if ihigh > len(hslices):
            nplots_in_last_row = len(hslices) % cols
            gs = GridSpec(rows,cols)
            ax = fig.add_subplot(gs[rows-1,:nplots_in_last_row])
        else:
            ax = fig.add_subplot(rows,1,r+1)
        ax.axesPatch.set_alpha(0.)
        ax.yaxis.set_visible(False)

        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        ax.spines['top'].set_position(('outward',5))

        if ilow < len(saxis.edges):
            if ihigh < len(saxis.edges):
                ax.set_xlim(saxis.edges[ilow],saxis.edges[ihigh])
                ax.xaxis.set_ticks(saxis.edges[ilow:ihigh+1])
            else:
                ax.set_xlim(saxis.edges[ilow],saxis.edges[-1])
        ax.set_xlabel(saxis.label, x=0.15)
        ax.xaxis.labelpad = -3

        # disable interactive pan/zoom for this axis
        ax.set_xlim = lambda *args,**kwargs: None

        ax.rasterize = False

        strip_axs += [ax]

    if hook is not None:
        return (fig,axs,strip_axs),hook_output
    else:
        return fig,axs,strip_axs

if __name__ == '__main__':
    from pyhep import Histogram

    d0 = (10, [0,100],'x')
    d1 = (10,[-0.5,100.5],'y')
    h2 = Histogram(d0,d1,'hist title','hist label')
    h2.fill(np.random.normal(100,50,10000),np.random.normal(50,10,10000))
    plothist_strip(h2)
    pyplot.show()
