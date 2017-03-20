from warnings import warn

try:
    from . import histogram_mpl
    have_mpl = True
    
    from matplotlib import pyplot

    '''
    Might try to find a useable backend here
    for convenience. Something like the following:
    
    for backend in matplotlib.rcsetup.interactive_bk:
        try:
            pyplot.switch_backend(backend)
            fig = pyplot.figure()
            pyplot.close(fig)
            print('backend: '+backend)
            break
        except ImportError as e:
            #print(e)
            continue
        except AttributeError as e:
            #print(e)
            continue
    '''

    def plothist_mpl(hist, **kwargs):
        opts = dict(subplot_kw = kwargs.pop('subplot_kw',None))
        opts.update(kwargs.pop('fig_kw',{}))
        fig,ax = pyplot.subplots(**opts)
        ret = [fig,ax]
        ret.append(ax.plothist(hist,**kwargs))
        return ret
        
except ImportError as e:
    print(e)
    warn('Could not import matplotlib.', ImportWarning)
    have_mpl = False

try:
    from . import histogram_bokeh
    have_bokeh = True
    
    from bokeh import plotting as bokeh
    from tempfile import TemporaryFile

    def plothist_bokeh(hist, **kwargs):
        bokeh.output_file(TemporaryFile(suffix='.html'))
        fig_kwargs = ['width','height']
        fig_kw = {k:kwargs.pop(k,None) for k in fig_kwargs}
        fig = bokeh.figure(**fig_kw)
        pt = fig.plothist(hist,**kwargs)
        return fig, pt
        
except ImportError:
    warn('Could not import bokeh.', ImportWarning)
    have_bokeh = False

if have_mpl:
    plothist = plothist_mpl
elif have_bokeh:
    plothist = plothist_bokeh
