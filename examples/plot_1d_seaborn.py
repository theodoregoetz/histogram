'''
This is the same as advanced_1d.py but using
seaborn to override much of the style of matplotlib.
'''

import numpy as np
from matplotlib import pyplot
from histogram import Histogram

import seaborn as sns

sns.set(palette="deep")

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
# method. (Axes.errorbar() in this case)
pttot = ax.plothist(htot, style='errorbar', lw=2,
                    capsize=4, markeredgewidth=2,
                    zorder=2)

# Histogram.smooth() applies a gaussian
# filter to the data
hsmooth = htot.copy(dtype=np.float64)
hsmooth.smooth(weight=1)
ax.plothist(hsmooth, style='line', lw=3, zorder=1)

pyplot.show()
