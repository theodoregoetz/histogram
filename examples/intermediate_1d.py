'''
Plotting a 1D histogram of some random data
in three different ways and including labels
on the x-axis, y-axis and a title.
'''

import numpy as np
from matplotlib import pyplot
from histogram import Histogram

np.random.seed(1)

h = Histogram(30, [0,10], 'x (cm)', 'counts', 'Random Data')
h.fill(np.random.normal(5,1,1000))

fig,ax = pyplot.subplots(1,3, figsize=(10,3))
fig.subplots_adjust(left=.08,bottom=.17,right=.98,top=.88,wspace=.37)

pt0 = ax[0].plothist(h, style='polygon', color='steelblue')
pt1 = ax[1].plothist(h, style='errorbar', color='olive')
pt3 = ax[2].plothist(h, style='line', color='darkred', lw=2)

pyplot.show()
