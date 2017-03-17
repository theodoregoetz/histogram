# coding: utf-8
'''
Plotting a 2D histogram of some random data
in three different ways and including labels
on the x-axis, y-axis and a title.
'''

from __future__ import unicode_literals

import numpy as np
from matplotlib import pyplot, rc, cm
from histogram import Histogram

rc('font', family=['Liberation Serif', 'DejaVu Serif', 'Arial'])

np.random.seed(1)

dx = (80, [0,10], 'x (cm)')
dy = (40, [-5,5], 'ε (J/cm$^2$)')
h = Histogram(dx, dy, 'counts', 'Random Data')

npoints = 10000
datax = np.random.normal(5,1,npoints)
datay = np.random.uniform(-5,5,npoints)
h.fill(datax, datay)

fig,ax = pyplot.subplots(1,3, figsize=(10,3))
fig.subplots_adjust(left=.06,bottom=.17,right=.95,top=.88,wspace=.52)

pt0 = ax[0].plothist(h, cmap=cm.GnBu)

pt1 = ax[1].plothist(h, cmap=cm.GnBu, style='contour', filled=True)

pt3a = ax[2].plothist(h, cmap=cm.GnBu)

h.smooth(1)
pt3b = ax[2].plothist(h, overlay=True,
                      style='contour', colors='black', levels=3)

pyplot.show()
