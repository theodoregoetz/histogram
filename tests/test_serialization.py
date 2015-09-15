# -*- coding: utf-8 -*-

import unittest

import sys
from copy import copy
import platform
import numpy as np

from histogram import Histogram, rc
from histogram.detail.strings import isstr, encoded_str, encode_dict, \
                                     decoded_str, decode_dict

class TestSerialization(unittest.TestCase):
    def setUp(self):
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

        self.h = h

    def test_npz(self):
        h = self.h.clone()
        filename = 'h'
        h.save(filename)
        hh = Histogram.load(filename)
        assert h.isidentical(hh)


    def test_hdf5(self):
        try:
            h = self.h.clone()
            import h5py
            filename = 'h.hdf5'
            h.save(filename)
            hh = Histogram.load(filename)
            assert h.isidentical(hh)
        except ImportError:
            pass

    def test_root(self):
        # For CERN/ROOT, we resorted to converting everything
        # into float64's so histograms are not typically
        # "identical" but they should be "close"
        try:
            h = self.h.clone()
            import ROOT
            filename = 'h.root'
            self.h.save(filename)
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

    def test_strings(self):
        assert isstr('a')
        assert isstr(r'a')
        assert isstr(u'a')
        assert isstr(b'a')

        s = 'α'
        es = r'\u03b1'
        assert encoded_str(s) == es
        assert s == decoded_str(es)

        d = {'a':'α'}
        ed = {'a':r'\u03b1'}

        encoded_d_copy = copy(d)
        encode_dict(encoded_d_copy)

        decoded_ed_copy = copy(ed)
        decode_dict(decoded_ed_copy)

        assert encoded_d_copy == ed
        assert decoded_ed_copy == d

        assert encoded_str(None) is None
        assert encoded_str(b'a') == 'a'

if __name__ == '__main__':
    unittest.main()
