import numpy as np
from matplotlib import pyplot
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..histogram import Histogram
from ..functions import gauss
from ..round import roundstr

from .cmap import flame

def make_axr(ax, overlaycolor='darkred', size='23%'):
    axr = ax.twinx()
    div = make_axes_locatable(axr)
    div.append_axes('right', size=size, add_to_figure=False)
    ax.spines['right'].set_color(overlaycolor)
    axr.tick_params(axis='y', colors=overlaycolor)
    axr.yaxis.label.set_color(overlaycolor)
    axr.yaxis.get_offset_text().set_color(overlaycolor)

    div = make_axes_locatable(ax)
    div.append_axes('right', size=size, add_to_figure=False)

    return axr

Axes.make_axr = make_axr


def make_axt(ax, overlaycolor='darkred', size='10%'):
    axt = ax.twiny()
    div = make_axes_locatable(axt)
    div.append_axes('right', size=size, add_to_figure=False)
    axt.tick_params(axis='x', colors=overlaycolor)

    return axt

Axes.make_axt = make_axt


def plothist_overlay(ax,hist,**kwargs):
    kw = dict(
        style='polygon',
        linewidth=2,
        edgecolor='darkred',
        facecolor='none')
    kw.update(**kwargs)
    return ax.plothist(hist,**kw)

Axes.plothist_overlay = plothist_overlay

def plothist_profile(ax,hist,mean,width,axis=0,**kwargs):
    markercolor = kwargs.pop('markercolor','black')
    overlaycolor = kwargs.pop('overlaycolor','darkred')

    fnprof = gauss
    if 'fnprof' in kwargs:
        fnprof = kwargs['fnprof']

    res = hist.profile_gauss(mean,width,axis=axis,**kwargs)
    if fnprof == None:
        res_slices, hprof = res
    else:
        res_slices, hprof, res_prof, stats_prof = res

    xx,dx,yy,dy = res_slices

    pt0,cb = ax.plothist(hist, alpha=0.5)

    axlim_x = ax.get_xlim()
    axlim_y = ax.get_ylim()

    pt1 = ax.errorbar(xx,xerr=dx,y=yy,yerr=dy,
        color=markercolor, linestyle='none', capsize=0)

    if axis == 0:
        ax1 = ax.make_axr(overlaycolor)
        pt2 = ax1.plothist_overlay(hprof)
    elif axis == 1:
        ax1 = ax.make_axt(overlaycolor)
        pt2 = ax1.plothist_overlay(hprof, baseline='left')
        #ax1.set_xlim(0, max(1.1*hprof.max(),1))

    ax.set_xlim(*axlim_x)
    ax.set_ylim(*axlim_y)

    if fnprof != None and res_prof != None:
        popt,pcov,ptest = res_prof
        ampsum,ampsumerr,sigmean,sigwid = stats_prof

        xmin,xmax = hist.axes[axis].range()
        xxfit = np.linspace(xmin,xmax,300)
        yyfit = fnprof(xxfit,*popt)

        if axis == 1:
            xxfit,yyfit = yyfit,xxfit

        ax1.plot(xxfit,yyfit,color=overlaycolor)

        if axis == 0:
            textcoordx = .05
            textha = 'left'
        if axis == 1:
            textcoordx = .95
            textha = 'right'

        ax.multi_text([
            r'$N = {} \pm {}$'.format(int(ampsum),int(ampsumerr)),
            r'$\mu = {}$'.format(roundstr(sigmean,ndigits=3,tex=True)),
            r'$\sigma = {}$'.format(roundstr(sigwid,ndigits=3,tex=True)),
        ],
            x=textcoordx,
            color=overlaycolor,
            horizontalalignment=textha)

    pts = pt0,cb,pt1,pt2
    return res,ax1,pts

Axes.plothist_profile = plothist_profile

def plothist_profile(*args,**kwargs):
    figsize = kwargs.pop('figsize',(6,4))
    fig = pyplot.figure(figsize=figsize)
    fig.subplots_adjust(bottom=.13)
    ax = fig.add_subplot(1,1,1)
    pts,cb = ax.plothist_profile(*args,**kwargs)
    return fig,ax,pts,cb
