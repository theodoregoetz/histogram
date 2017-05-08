import matplotlib
matplotlib.use('GTK3Agg')

from numpy import random as rand
from matplotlib import pyplot
from histogram import Histogram

rand.seed(1)

npoints = 10000
xdata = rand.normal(100,50,npoints)
ydata = rand.normal(50,10,npoints)

d0 = (10, [0,100],'$x$')
d1 = (10,[-0.5,100.5],'$y$')
h2 = Histogram(d0,d1,'$z$','Random Data')
h2.fill(xdata,ydata)

h2slices = list(h2.slices())
axslices = h2.axes[0]

fig = pyplot.figure(figsize=(12,12))
axs,saxs = fig.plothist_strip(h2slices,axslices)

pyplot.show()
