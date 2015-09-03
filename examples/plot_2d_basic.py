'''
Simple 2D histogram example.
'''

import numpy as np
from matplotlib import pyplot, cm
from histogram import Histogram, plothist

np.random.seed(1)

# 2D histogram with 30x40 bins
h = Histogram(30,[0,10],40,[-5,5])

# filling the histogram with some random data
npoints = 100000
datax = np.random.normal(5,1,npoints)
datay = np.random.uniform(-5,5,npoints)
h.fill(datax,datay)

# filling with even more data
datax = np.random.uniform(0,10,npoints)
datay = np.random.normal(0,1,npoints)
h.fill(datax,datay)

# using the plothist() convenience method
fig,ax,pt = plothist(h, cmap=cm.Blues)

# display figure to screen
pyplot.show()
