import numpy as np

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

def subplots(fig, nrows=1, ncols=1, sharex=False, sharey=False,
             squeeze=True, subplot_kw=None, gridspec_kw=None):
    """
    This is the same as matplotlib.pyplot.subplots but for a
    pre-existing matplotlib.figure.Figure object
    """
    # for backwards compatibility
    if isinstance(sharex, bool):
        if sharex:
            sharex = "all"
        else:
            sharex = "none"
    if isinstance(sharey, bool):
        if sharey:
            sharey = "all"
        else:
            sharey = "none"
    share_values = ["all", "row", "col", "none"]
    if sharex not in share_values:
        # This check was added because it is very easy to type subplots(1, 2, 1)
        # when subplot(1, 2, 1) was intended. In most cases, no error will
        # ever occur, but mysterious behavior will result because what was
        # intended to be the subplot index is instead treated as a bool for
        # sharex.
        if isinstance(sharex, int):
            warnings.warn("sharex argument to subplots() was an integer."
                          " Did you intend to use subplot() (without 's')?")

        raise ValueError("sharex [%s] must be one of %s" % \
                (sharex, share_values))
    if sharey not in share_values:
        raise ValueError("sharey [%s] must be one of %s" % \
                (sharey, share_values))
    if subplot_kw is None:
        subplot_kw = {}
    if gridspec_kw is None:
        gridspec_kw = {}

    gs = GridSpec(nrows, ncols, **gridspec_kw)

    # Create empty object array to hold all axes.  It's easiest to make it 1-d
    # so we can just append subplots upon creation, and then
    nplots = nrows*ncols
    axarr = np.empty(nplots, dtype=object)

    # Create first subplot separately, so we can share it if requested
    ax0 = fig.add_subplot(gs[0, 0], **subplot_kw)
    #if sharex:
    #    subplot_kw['sharex'] = ax0
    #if sharey:
    #    subplot_kw['sharey'] = ax0
    axarr[0] = ax0

    r, c = np.mgrid[:nrows, :ncols]
    r = r.flatten() * ncols
    c = c.flatten()
    lookup = {
            "none": np.arange(nplots),
            "all": np.zeros(nplots, dtype=int),
            "row": r,
            "col": c,
            }
    sxs = lookup[sharex]
    sys = lookup[sharey]

    # Note off-by-one counting because add_subplot uses the MATLAB 1-based
    # convention.
    for i in range(1, nplots):
        if sxs[i] == i:
            subplot_kw['sharex'] = None
        else:
            subplot_kw['sharex'] = axarr[sxs[i]]
        if sys[i] == i:
            subplot_kw['sharey'] = None
        else:
            subplot_kw['sharey'] = axarr[sys[i]]
        axarr[i] = fig.add_subplot(gs[i // ncols, i % ncols], **subplot_kw)

    # returned axis array will be always 2-d, even if nrows=ncols=1
    axarr = axarr.reshape(nrows, ncols)

    # turn off redundant tick labeling
    if sharex in ["col", "all"] and nrows > 1:
    #if sharex and nrows>1:
        # turn off all but the bottom row
        for ax in axarr[:-1, :].flat:
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)

    if sharey in ["row", "all"] and ncols > 1:
    #if sharey and ncols>1:
        # turn off all but the first column
        for ax in axarr[:, 1:].flat:
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)

    if squeeze:
        # Reshape the array to have the final desired dimension (nrow,ncol),
        # though discarding unneeded dimensions that equal 1.  If we only have
        # one subplot, just return it instead of a 1-element array.
        if nplots==1:
            ret = axarr[0,0]
        else:
            ret = axarr.squeeze()
    else:
        # returned axis array will be always 2-d, even if nrows=ncols=1
        ret = axarr.reshape(nrows, ncols)

    return ret

Figure.subplots = subplots
