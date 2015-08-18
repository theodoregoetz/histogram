from matplotlib import pyplot

from .histogram import plothist
#from .histogram_grid import plothist_grid
#from .histogram_strip import plothist_strip
#from .histogram_profile import make_axr, plothist_profile

def plothist(hist, **kwargs):
    fig,ax = pyplot.subplots()
    ret = [fig,ax]
    ret.append(ax.plothist(hist,**kwargs))
    return ret
