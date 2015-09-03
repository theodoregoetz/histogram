from warnings import warn

try:
    from . import histogram_mpl
    have_mpl = True
except ImportError:
    warn('Could not import matplotlib.', ImportWarning)
    have_mpl = False

try:
    from . import histogram_bokeh
    have_bokeh = True
except ImportError:
    warn('Could not import bokeh.', ImportWarning)
    have_bokeh = False

if have_mpl:
    from matplotlib import pyplot

    def plothist(hist, **kwargs):
        opts = dict(subplot_kw = kwargs.pop('subplot_kw',None))
        opts.update(kwargs.pop('fig_kw',{}))
        fig,ax = pyplot.subplots(**opts)
        ret = [fig,ax]
        ret.append(ax.plothist(hist,**kwargs))
        return ret

elif have_bokeh:
    from bokeh import plotting as bokeh
    from tempfile import TemporaryFile

    def plothist(hist, **kwargs):
        bokeh.output_file(TemporaryFile(suffix='.html'))
        fig_kwargs = ['width','height']
        fig_kw = {k:kwargs.pop(k,None) for k in fig_kwargs}
        fig = bokeh.figure(**fig_kw)
        pt = fig.plothist(hist,**kwargs)
        return fig, pt
