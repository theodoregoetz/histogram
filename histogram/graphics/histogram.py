from warnings import warn
from tempfile import TemporaryFile

try:
    from matplotlib import pyplot
    from . import histogram_mpl
    have_mpl = True
except ImportError:
    warn('Could not import matplotlib.', ImportWarning)
    have_mpl = False

try:
    from bokeh import plotting as bokeh
    from . import histogram_bokeh
    have_bokeh = True
except ImportError:
    warn('Could not import bokeh.', ImportWarning)
    have_bokeh = False

def plothist(hist, **kwargs):
    global have_mpl
    global have_bokeh

    if have_mpl:
        opts = dict(
            subplot_kw = kwargs.pop('subplot_kw',None),
            fig_kw = kwargs.pop('fig_kw',None))
        fig,ax = pyplot.subplots(**opts)
        ret = [fig,ax]
        ret.append(ax.plothist(hist,**kwargs))
        return ret
    elif have_bokeh:
        bokeh.output_file(TemporaryFile(suffix='.html'))
        fig_kwargs = ['width','height']
        fig_kw = {k:kwargs.pop(k,None) for k in fig_kwargs}
        fig = bokeh.figure(**fig_kw)
        pt = fig.plothist(hist,**kwargs)
        return fig
