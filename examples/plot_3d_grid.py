from numpy import random as rand
from matplotlib import pyplot, cm
from histogram import Histogram

npoints = 1000000
datax = rand.normal(100,40,npoints)
datay = rand.normal(100,60,npoints)
dataz = rand.normal(50,20,npoints)

d0 = (10,[0,100],'x')
d1 = (9, [0,100],'y')
d2 = (100,[0,100],'z')
h3 = Histogram(d0,d1,d2,'counts','Random Data')

h3.fill(datax,datay,dataz)

fig = pyplot.figure(figsize=(8,8))
axs,axtot,axins = fig.plothist_grid(h3, ymin=0, cmap=cm.viridis) #copper_r)

pyplot.show()
