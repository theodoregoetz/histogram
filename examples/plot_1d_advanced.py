'''
Overlaying three random-normal distributions on top
of eachother, using the color cycle defined through
`matplotlib.rc`.

The sum of the histograms is plotted using the errorbar
style. This histogram is then smoothed using a Gaussian
filter and plotted as a line.
'''

from cycler import cycler
import numpy as np
from matplotlib import pyplot
from histogram import Histogram

from matplotlib import rc as mplrc
from histogram import rc as histrc

# interesting colors
mplrc('axes', prop_cycle=cycler('color', ['steelblue','olive','darkred',
                                          'darkorchid','goldenrod','black']))

# default is alpha=0.6, but we want slightly lighter
# patches for overlaying many histograms
histrc.plot.patch.alpha = 0.5

np.random.seed(1)

hh = []
for mean in [2,5,7]:
    hh.append(Histogram(30, [0,10], 'x (cm)', 'counts', 'Random Data'))
    hh[-1].fill(np.random.normal(mean,1,1000))

fig,ax = pyplot.subplots()

pts = []
for h in hh:
    pts.append(ax.plothist(h, zorder=-1))

# overlay option turns autoscale off
# before plotting and adjusts alpha
# of the patches. Here, we want to
# readjust the scale for the sum of
# all histograms
htot = sum(hh)
ax.set_autoscale_on(True)

# all kwargs unknown by histogram will
# be passed onto the underlying plotting
# method Axes.errorbar() in this case
pttot = ax.plothist(htot, style='errorbar', lw=2,
                    capsize=4, markeredgewidth=2,
                    zorder=2)

# Histogram.smooth() applies a gaussian
# filter to the data
hsmooth = htot.copy(dtype=np.float64)
hsmooth.smooth(weight=1)
ax.plothist(hsmooth, style='line', lw=3, zorder=1)

pyplot.show()
