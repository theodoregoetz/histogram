import math
import numpy as np

import matplotlib
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator, LogLocator
from matplotlib.colors import LogNorm

from . import (histogram_mpl_strip, histogram_mpl_grid,
               histogram_mpl_profile)
from .. import rc

def plothist_errorbar(ax, hist, **kwargs):
    baseline = kwargs.pop('baseline',rc.plot.baseline)
    mask = kwargs.pop('mask',None)
    kw = dict(
        linestyle = kwargs.pop('linestyle','none') )
    kw.update(kwargs)

    if not hist.has_uncert:
        hist.uncert = np.sqrt(np.abs(hist.data))

    if mask is not None:
        (x,),y = hist.grid()[mask], hist.data[mask]
        xerr,yerr = [e[mask] for e in hist.errorbars()]
    else:
        (x,),y = hist.grid(), hist.data
        xerr,yerr = hist.errorbars()

    nans = ~np.isfinite(yerr)
    if np.any(nans):
        yerr[nans] = 0

    mask = np.isfinite(y)
    if not np.all(mask):
        x = x[mask]
        y = y[mask]
        xerr = xerr[mask]
        yerr = yerr[mask]

    if baseline == 'left':
        y,x = x,y
        xerr,yerr = yerr,xerr

    return ax.errorbar(x, y, xerr=xerr, yerr=yerr, **kw)

def plothist_polygon(ax, hist, **kwargs):
    baseline = kwargs.pop('baseline',rc.plot.baseline)
    ymin = kwargs.pop('ymin', 0)

    kw = dict(
        linewidth = kwargs.pop('linewidth',kwargs.pop('lw',0)),
        alpha = kwargs.pop('alpha',rc.plot.patch.alpha))
    if not any([x in kwargs for x in ['facecolor','edgecolor']]):
        kw['color'] = kwargs.pop('color',next(ax._get_lines.prop_cycler)['color'])
    kw.update(kwargs)

    pt = ax.fill_between(hist.axes[0].edges,
                         np.concatenate((hist.data, [0])),
                         y2=ymin,
                         step='post',
                         **kw)

    return pt

def plothist_line(ax, hist, **kwargs):
    baseline = kwargs.pop('baseline',rc.plot.baseline)

    kw = dict(
        linewidth = kwargs.pop('linewidth',kwargs.pop('lw',1)),
        alpha = kwargs.pop('alpha',1),
        color = kwargs.pop('color',next(ax._get_lines.prop_cycler)['color']) )
    kw.update(kwargs)

    x, y, extent = hist.asline()

    if baseline == 'left':
        x,y = y,x

    return ax.plot(x,y,**kw)

def plothist_1d(ax, hist, **kwargs):
    baseline = kwargs.get('baseline',rc.plot.baseline)

    overlay = kwargs.pop('overlay',False)
    yscale = kwargs.pop('yscale',None)
    style = kwargs.pop('style',None)
    ymin = kwargs.pop('ymin',None)

    if overlay:
        ax.set_autoscale_on(False)

    if kwargs.get('mask',None) is not None:
        if style is None:
            style = 'errorbar'

    if style is None:
        if hist.has_uncert:
            style = 'errorbar'
        else:
            maxratios = np.array([x.max() for x in hist.errorbars(asratio=True)])
            if np.any(maxratios > 0.2):
                style = 'errorbar'
            else:
                style = 'polygon'

    extent_kwargs = dict(
        pad = kwargs.pop('pad',[0,0,0,0.05]),
        uncert = (style == 'errorbar'))

    if style == 'errorbar':
        pt = plothist_errorbar(ax, hist, **kwargs)
    elif style == 'line':
        pt = plothist_line(ax, hist, **kwargs)
    else:
        pt = plothist_polygon(ax, hist, **kwargs)

    if not overlay:
        if hist.axes[0].label is not None:
            if baseline == 'left':
                ax.set_ylabel(hist.axes[0].label)
            elif baseline == 'bottom':
                ax.set_xlabel(hist.axes[0].label)
        if hist.label is not None:
            if baseline == 'left':
                ax.set_xlabel(hist.label)
            elif baseline == 'bottom':
                ax.set_ylabel(hist.label)

        autox = ax.get_autoscalex_on()
        autoy = ax.get_autoscaley_on()

        if autox or autoy:

            hmin = hist.min()
            if not np.isclose(hmin, 0):
                extent_kwargs['pad'][2] = 0.05

            xmin,xmax,y0,ymax = hist.extent(**extent_kwargs)

            if yscale == 'log':
                ax.set_yscale('log', nonposy='clip')
                data_min = np.min(hist.data[np.where(hist.data>0)])
                minexp = int(math.log(data_min, 10))
                maxexp = int(math.log(np.max(hist.data), 10)+0.2)
                y0,ymax = 10**(minexp-1), 10**(maxexp+1)

            if ymin is None:
                ymin = y0

            if baseline == 'left':
                xmin,xmax,ymin,ymax = ymin,ymax,xmin,xmax

            if autoy:
                ax.set_ylim(ymin, ymax)

            if autox:
                ax.set_xlim(xmin, xmax)

    return pt

def plothist_imshow(ax, hist, **kwargs):
    kw = dict(
        origin = 'lower',
        aspect = 'auto',
        interpolation = 'nearest',
        extent = hist.extent(2) )
    kw.update(kwargs)
    return ax.imshow(hist.data.T, **kw)

def plothist_pcolor(ax, hist, **kwargs):
    overlay = kwargs.pop('overlay',False)
    extent = hist.extent(2)
    xx,yy = hist.edge_grid

    ret = ax.pcolor(xx, yy, hist.data, **kwargs)

    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])

    return ret

def plothist_contour(ax, hist, **kwargs):
    levels = kwargs.pop('levels',None)
    filled = kwargs.pop('filled',False)
    args = list(hist.grid())
    args.append(hist.data)
    if levels is not None:
        args.append(levels)
    if filled:
        return ax.contourf(*args, **kwargs)
    else:
        return ax.contour(*args, **kwargs)

def add_colorbar(ax, plt, hist, **kwargs):
    scale = kwargs.pop('scale',None)
    kw = dict(vmin=None)
    kw.update(kwargs)

    try:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.1, pad=0.1)
        cb = ax.figure.colorbar(plt, cax=cax, **kwargs)

    except ImportError:
        cb = ax.figure.colorbar(plt, ax=ax, **kwargs)

    if scale == 'log':
        zlim = cb.get_clim()
        zmin = int(math.log(zlim[0], 10))
        zmax = int(math.log(zlim[1], 10))
        dz = zmax - zmin
        if dz < 2:
            cb.locator = LogLocator(subs = [1,2,3,4,5,6,7,8,9])
        elif dz < 4:
            cb.locator = LogLocator(subs = [1,2,4,6,8])
        else:
            cb.locator = LogLocator()

    else:
        cb.formatter.set_scientific(True)
        if hist.data.dtype in [np.int64, np.int32]:
            if cb.ax.get_xlim()[-1] < 1000:
                cb.locator = MaxNLocator(integer=True)

    cb.update_ticks()

    if hist.label is not None:
        cb.set_label(hist.label)

    return cb

def plothist_2d(ax, hist, **kwargs):
    overlay = kwargs.pop('overlay',False)
    if overlay:
        ax.set_autoscale_on(False)

    kw_cb = dict(scale = kwargs.pop('zscale',None))

    if kw_cb['scale'] == 'log':
        kwargs['norm'] = LogNorm()

    if kwargs.pop('style',None) == 'contour':
        pt = plothist_contour(ax, hist, **kwargs)
    else:
        if hist.isuniform():
            pt = plothist_imshow(ax, hist, **kwargs)
        else:
            pt = plothist_pcolor(ax, hist, **kwargs)

    if not overlay:

        # if bin centers are close to integer values
        # then limit the axes' ticks to integers
        for hax,axis in zip(hist.axes,[ax.xaxis,ax.yaxis]):
            grid = hax.bincenters()
            f = grid - np.floor(grid)
            median = np.median(f)
            if abs(median) < 0.0001:
                maxdev = np.max(np.abs(f - median))
                if maxdev < 0.0001:
                    axis.set_major_locator(MaxNLocator(integer=True))

        try:
            cb = add_colorbar(ax, pt, hist, **kw_cb)
        except AttributeError:
            cb = None

        if hist.axes[0].label is not None:
            ax.set_xlabel(hist.axes[0].label)
        if hist.axes[1].label is not None:
            ax.set_ylabel(hist.axes[1].label)

    if overlay:
        return pt
    else:
        return pt,cb

def plothist(ax, hist, **kwargs):
    if hist.dim == 1:
        ret = plothist_1d(ax, hist, **kwargs)
    elif hist.dim == 2:
        ret = plothist_2d(ax, hist, **kwargs)
    else:
        raise NotImplementedError('Can not plot histograms of dim > 2.')

    if hist.title is not None:
        ax.set_title(hist.title)

    return ret

matplotlib.axes.Axes.plothist = plothist
