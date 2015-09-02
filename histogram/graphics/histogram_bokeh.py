from bokeh import plotting as bokeh

from .. import rc

def plothist_polygon(fig, hist, **kwargs):
    baseline = kwargs.pop('baseline',rc.plot.baseline)
    polygon_kwargs = dict(
        ymin = kwargs.pop('ymin',0),
        xlow = kwargs.pop('xlow',None),
        xhigh = kwargs.pop('xhigh',None) )
    points,extent = hist.aspolygon()
    if baseline == 'left':
        y,x = zip(*points)
    else:
        x,y = zip(*points)

    pt = fig.patch(x=x,y=y,**kwargs)
    return pt

def plothist_1d(fig, hist, **kwargs):
    return plothist_polygon(fig, hist, **kwargs)

def plothist(fig, hist, **kwargs):
    if hist.dim == 1:
        return plothist_1d(fig, hist, **kwargs)
    else:
        raise NotImplementedError()

bokeh.Figure.plothist = plothist
