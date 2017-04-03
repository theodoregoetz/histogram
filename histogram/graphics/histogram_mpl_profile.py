import numpy as np
from scipy import stats

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .. import Histogram

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

def make_axt(ax, overlaycolor='darkred', size='10%'):
    axt = ax.twiny()
    div = make_axes_locatable(axt)
    div.append_axes('right', size=size, add_to_figure=False)
    axt.tick_params(axis='x', colors=overlaycolor)

    return axt

def plothist_profile(ax,hist,axis=0,**kwargs):
    assert hist.dim == 2

    markercolor = kwargs.pop('markercolor','black')
    overlaycolor = kwargs.pop('overlaycolor','darkred')
    cmap = kwargs.pop('cmap',None)

    pax = hist.axes[axis]
    fitax = hist.axes[({0,1} - {axis}).pop()]
    mid0 = fitax.mid
    sig0 = 0.2*(fitax.max-fitax.min)

    def _fn(x,n,m,s):
        return n * stats.norm(m,s).pdf(x)

    def _p0(h,mid0=mid0,sig0=sig0):
        return [h.integral()[0],mid0,sig0]

    fn = kwargs.pop('fn', _fn)
    p0 = kwargs.pop('p0', lambda h: _p0(h))

    hist_par = kwargs.pop('hist_par',0)
    marker_par = kwargs.pop('marker_par',1)
    marker_err_par = kwargs.pop('marker_err_par',2)

    hprof = Histogram(pax)
    yy = np.empty((pax.nbins,))
    dy = np.empty((pax.nbins,))
    fit_slices_results = []
    for i,h in enumerate(hist.slices(axis)):
        fit_slices_results.append(h.fit(fn,p0,**kwargs))
        popt,pcov,ptest = fit_slices_results[-1]
        hprof.data[i] = popt[hist_par]
        yy[i] = popt[marker_par]
        dy[i] = popt[marker_err_par]

    pt0,cb = ax.plothist(hist, alpha=0.5, cmap=cmap)

    axlim_x = ax.get_xlim()
    axlim_y = ax.get_ylim()

    pt1 = ax.errorbar(pax.bincenters(),xerr=0.5*pax.binwidths(),
        y=yy,yerr=dy,
        color=markercolor, linestyle='none', capsize=0)

    kw = dict(
        style='polygon',
        linewidth=2,
        edgecolor=overlaycolor,
        facecolor='none')

    if axis == 0:
        ax1 = make_axr(ax, overlaycolor)
        pt2 = ax1.plothist(hprof, **kw)
    elif axis == 1:
        ax1 = ax.make_axt(overlaycolor)
        pt2 = ax1.plothist(hprof, baseline='left', **kw)
        #ax1.set_xlim(0, max(1.1*hprof.max(),1))

    ax.set_xlim(*axlim_x)
    ax.set_ylim(*axlim_y)

    fnprof = kwargs.pop('fnprof', _fn)
    p0prof = kwargs.pop('p0prof', lambda h: _p0(h))
    if fnprof is not None:

        fit_prof_results = hprof.fit(fnprof,p0prof)
        popt,pcov,ptest = fit_prof_results

        xmin,xmax = hprof.axes[0].limits
        xxfit = np.linspace(xmin,xmax,300)
        yyfit = fnprof(xxfit,*popt)

        if axis == 1:
            xxfit,yyfit = yyfit,xxfit

        pt3 = ax1.plot(xxfit,yyfit,color=overlaycolor)

    return hprof, fit_slices_results, fit_prof_results

matplotlib.axes.Axes.plothist_profile = plothist_profile
