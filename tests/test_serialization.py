# -*- coding: utf-8 -*-

import sys
import platform
import numpy as np

from histogram import Histogram, rc

rc.overwrite.overwrite = 'always'

np.random.seed(1)

h = Histogram(100,[0,10],'Δx', 'y', 'title')
h.fill(np.random.normal(5,2,10000))
h.uncert = np.sqrt(h.data)

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

filename = 'h'
h.save(filename)
hh = Histogram.load(filename)
assert h.isidentical(hh)


try:
    import h5py
    filename = 'h.hdf5'
    h.save(filename)
    hh = Histogram.load(filename)
    assert h.isidentical(hh)
except ImportError:
    pass


# For CERN/ROOT, we resorted to converting everything
# into float64's so histograms are not typically
# "identical" but they should be "close"

try:
    import ROOT
    filename = 'h.root'
    h.save(filename)
    hh = Histogram.load(filename)
    assert np.allclose(h.data,hh.data)
    assert np.allclose(h.uncert,hh.uncert)
    assert h.label == hh.label
    assert h.title == hh.title
    for a,aa in zip(h.axes,hh.axes):
        assert np.allclose(a.edges,aa.edges)
        assert a.label == aa.label
except ImportError:
    pass
