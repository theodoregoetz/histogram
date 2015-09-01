import numpy as np

from histogram import Histogram

np.random.seed(1)

h = Histogram(100,[0,10],'Δx', 'y', 'title')
h.fill(np.random.normal(5,2,10000))
