import sys
import numpy as np

from histogram import Histogram
from data import h

if sys.version_info < (3,0):
    def _to_unicode(s):
        if not isinstance(s,unicode):
            return unicode(s,'utf-8')
        else:
            return s
    h.title = _to_unicode(h.title)
    h.label = _to_unicode(h.label)
    for ax in h.axes:
        ax.label = _to_unicode(ax.label)


infiles = ['h2','h3','h2.hdf5','h3.hdf5']

for infile in infiles:
    print(infile)
    hh = Histogram.load(infile)
    assert h.isidentical(hh)


# For CERN/ROOT, we resorted to converting everything
# into float64's so histograms are not typically
# "identical" but they should be "close"
infiles = ['h2.root','h3.root']

for infile in infiles:
    print(infile)
    hh = Histogram.load(infile)
    assert np.allclose(h.data,hh.data)
    assert np.allclose(h.uncert,hh.uncert)
    assert h.label == hh.label
    assert h.title == hh.title
    for a,aa in zip(h.axes,hh.axes):
        assert np.allclose(a.edges,aa.edges)
        assert a.label == aa.label
