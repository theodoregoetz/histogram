from bokeh import plotting as bokeh

from .. import rc

def plothist_polygon(fig, hist, **kwargs):
    baseline = kwargs.pop('baseline',rc.plot.baseline)
    polygon_kwargs = dict(
        ymin = kwargs.pop('ymin',0),
        xlow = kwargs.pop('xlow',None),
        xhigh = kwargs.pop('xhigh',None) )

    x,y,extent = hist.aspolygon(**polygon_kwargs)

    if baseline == 'left':
        y,x = x,y

    pt = fig.patch(x=x,y=y,**kwargs)
    return pt

def plothist_line(fig, hist, **kwargs):
    baseline = kwargs.pop('baseline',rc.plot.baseline)
    polygon_kwargs = dict(
        ymin = kwargs.pop('ymin',0),
        xlow = kwargs.pop('xlow',None),
        xhigh = kwargs.pop('xhigh',None) )

    a,y,extent = hist.aspolygon()

    if baseline == 'left':
        y,x = x,y

    pt = fig.patch(x=x,y=y,**kwargs)
    return pt

def plothist_1d(fig, hist, **kwargs):
    return plothist_polygon(fig, hist, **kwargs)

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
