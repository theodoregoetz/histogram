'''
This demonstrated plotting a 1D histogram to
a bokeh figure which in this case is a temporary
html file that gets loaded up in a browser.
'''

import sys
import os
import numpy as np
from tempfile import NamedTemporaryFile
from bokeh import plotting as bokeh
from histogram import Histogram

if sys.version_info < (3,0):
    input = raw_input

np.random.seed(1)

h = Histogram(30,[0,10])
h.fill(np.random.normal(5,1,1000))

fout = NamedTemporaryFile(delete=False, prefix='hist', suffix='.html',
                          dir=os.getcwd())
fname = fout.name
fout.close()

bokeh.output_file(fname)
fig = bokeh.figure()
pt = fig.plothist(h)

bokeh.show(fig)
_ = input('press ENTER to exit.\n')
os.remove(fname)
