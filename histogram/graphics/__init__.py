from matplotlib import pyplot

from .histogram import plothist
#from .histogram_grid import plothist_grid
#from .histogram_strip import plothist_strip
#from .histogram_profile import make_axr, plothist_profile

def plothist(hist, **kwargs):
    opts = dict(
        subplot_kw = kwargs.pop('subplot_kw',None),
        fig_kw = kwargs.pop('fig_kw',None))
    fig,ax = pyplot.subplots(**opts)
    ret = [fig,ax]
    ret.append(ax.plothist(hist,**kwargs))
    return ret
