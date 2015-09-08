import numpy as np
from numpy import random as rand
from matplotlib import pyplot, cm
from histogram import Histogram

rand.seed(1)

npoints = 100000
xdata = rand.normal(100,50,npoints)
ydata = rand.normal(50,10,npoints)

d0 = (30, [0,100],'$x$')
d1 = (40,[-0.5,100.5],'$y$')
h2 = Histogram(d0,d1,'$z$','Random Data')
h2.fill(xdata,ydata)

fig,ax = pyplot.subplots()

hprof,fitslices,fitprof = ax.plothist_profile(h2, cmap=cm.Blues)

popt,pcov,ptest = fitprof
perr = np.sqrt(popt)

msg = '''\
$N = {opt[0]:.0f} \pm {err[0]:.0f}$
$\mu = {opt[1]:.1f} \pm {err[1]:.1f}$'''

ax.text(.05, .95,
    msg.format(opt=popt,err=perr),
    horizontalalignment = 'left',
    verticalalignment = 'top',
    bbox = dict(alpha = 0.7, color='white'),
    transform = ax.transAxes)

pyplot.show()
