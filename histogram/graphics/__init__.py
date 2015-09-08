from .histogram import plothist, have_mpl

if have_mpl:
    from .histogram_strip import plothist_strip
