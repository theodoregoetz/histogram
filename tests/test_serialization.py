# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import platform
import os
import numpy as np
from numpy import random as rand

from histogram import Histogram, rc

rand.seed(1)

homedir = os.environ['HOME']
pyvers = platform.python_version().split('.')[0]

rc.histdir = os.path.join(homedir,'.histogram','tests')
rc.overwrite.overwrite = 'always'

h = Histogram(100,(0,10),'Δx','y','title')
h.fill(rand.normal(5,2,10000))
h.uncert = np.sqrt(h.data)

h.save(pyvers+'/serialization_h')

hh = Histogram.load(pyvers+'/serialization_h')

compare(h,hh)

othervers = '2' if pyvers == '3' else '3'
hhh = Histogram.load(othervers+'/serialization_h')
compare(h,hhh)
