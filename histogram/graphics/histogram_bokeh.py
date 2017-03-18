import numpy as np

from bokeh import plotting as bokeh
from bokeh.models import Range1d, LogAxis

from .. import rc

def plothist_errorbar(fig, hist, **kwargs):
    baseline = kwargs.pop('baseline',rc.plot.baseline)
    mask = kwargs.pop('mask',None)

    if mask is not None:
        x,y = hist.grid()[0][mask], hist.data[mask]
        xerr,yerr = [e[mask] for e in hist.errorbars()]
    else:
        x,y = hist.grid()[0], hist.data
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

    x0 = np.concatenate((x - xerr, x))
    x1 = np.concatenate((x + xerr, x))
    y0 = np.concatenate((y, y - yerr))
    y1 = np.concatenate((y, y + yerr))

    return fig.segment(x0, y0, x1, y1, **kwargs)

def plothist_polygon(fig, hist, **kwargs):
    baseline = kwargs.pop('baseline',rc.plot.baseline)
    ymin = kwargs.pop('ymin',0)

    x,y,extent = hist.asline()

    x = np.concatenate(([x[0]], x, [x[-1]]))
    y = np.concatenate(([ymin], y, [ymin]))

    if baseline == 'left':
        y,x = x,y

    pt = fig.patch(x,y,**kwargs)
    return pt

def plothist_line(fig, hist, **kwargs):
    baseline = kwargs.pop('baseline',rc.plot.baseline)

    x,y,extent = hist.asline()

    pt = fig.line(x,y,**kwargs)
    return pt

def plothist_1d(fig, hist, **kwargs):
    baseline = kwargs.get('baseline',rc.plot.baseline)

    overlay = kwargs.pop('overlay',False)
    style = kwargs.pop('style',None)
    ymin = kwargs.pop('ymin',None)

    extent_kwargs = dict(pad = kwargs.pop('pad',[0,0,0,0.05]))

    if style == 'errorbar':
        pt = plothist_errorbar(fig, hist, **kwargs)
    elif style == 'line':
        pt = plothist_line(fig, hist, **kwargs)
    else:
        pt = plothist_polygon(fig, hist, **kwargs)

    if not overlay:

        hmin = hist.min()
        if not np.isclose(hmin,0):
            if (hmin > 0) and (hmin > 0.3 * abs(hist.max() - hmin)):
                extent_kwargs['pad'][2] = 0.05

        xmin,xmax,y0,ymax = hist.extent(**extent_kwargs)

        if ymin is None:
            ymin = y0

        if baseline == 'left':
            xmin,xmax,ymin,ymax = ymin,ymax,xmin,xmax

        #pt.x_range = Range1d(start=xmin, end=xmax)
        #pt.xaxis.bounds = xmin,xmax
        #pt.y_range = Range1d(start=ymin, end=ymax)
        #pt.yaxis.bounds = ymin,ymax

        #if hist.axes[0].label is not None:
        #    if baseline == 'left':
        #        pt.xaxis.axis_label = hist.axes[0].label
        #    elif baseline == 'bottom':
        #        pt.yaxis.axis_label = hist.axes[0].label
        #if hist.label is not None:
        #    if baseline == 'left':
        #        pt.xaxis.axis_label = hist.label
        #    elif baseline == 'bottom':
        #        pt.yaxis.axis_label = hist.label

    return pt

def plothist_2d(fig, hist, **kwargs):
    return plothist_(fig, hist, **kwargs)

def plothist(fig, hist, **kwargs):
    if hist.dim == 1:
        ret = plothist_1d(fig, hist, **kwargs)
    elif hist.dim == 2:
        ret = plothist_2d(fig, hist, **kwargs)
    else:
        raise NotImplementedError('Can not plot histograms of dim > 2.')

    if hist.title is not None:
        ax.set_title(hist.title)

    return ret

bokeh.Figure.plothist = plothist
