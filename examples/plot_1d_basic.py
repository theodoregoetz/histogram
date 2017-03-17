'''
About as basic an example as one can get.
'''

import numpy as np
from matplotlib import pyplot
from histogram import Histogram

np.random.seed(1)

h = Histogram(30,[0,10])
h.fill(np.random.normal(5,1,1000))

fig,ax = pyplot.subplots()
pt = ax.plothist(h, alpha=0.3)

pyplot.show()
