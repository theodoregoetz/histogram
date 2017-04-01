# coding: utf-8
from __future__ import unicode_literals

import numpy as np
import os
import unittest
import warnings

from tempfile import NamedTemporaryFile

from histogram import rc as histrc
from histogram import Histogram


try:
    import ROOT
    NO_PYROOT = False
except ImportError:
    NO_PYROOT = True


@unittest.skipIf(NO_PYROOT, 'no PyROOT found')
class TestSerializationRoot(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('always')
        histrc.overwrite.overwrite = 'always'

    def tearDown(self):
        histrc.overwrite.overwrite = 'ask'

    def test_unicode(self):
        ftmp = NamedTemporaryFile(suffix='.root', delete=False)
        try:
            ftmp.close()
            h = Histogram(3,[0,3])
            h.data[:] = [-3,0,5]
            h.uncert = np.sqrt(np.abs(h.data)) # ROOT will always return uncert
            h.title = 'χ-squared'
            h.label = 'αβγ'
            h.axes[0].label = 'θ'
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

        finally:
            os.remove(ftmp.name)

    def test_root_1d(self):
        ftmp = NamedTemporaryFile(suffix='.root', delete=False)
        try:
            ftmp.close()
            h = Histogram(3,[0,3])
            h.data[:] = [-3,0,5]
            h.uncert = np.sqrt(np.abs(h.data)) # ROOT will always return uncert
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

            h.axes[0].label = 'x (cm)'
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

            h.label = 'counts'
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

            h.title = 'title'
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

        finally:
            os.remove(ftmp.name)

    def test_root_2d(self):
        ftmp = NamedTemporaryFile(suffix='.root', delete=False)
        try:
            ftmp.close()
            h = Histogram(3,[0,3],4,[0,4])
            h.data[:] = [[-3,0,5,3],[-2,0,4,2],[-1,0,3,1024]]
            h.uncert = np.sqrt(np.abs(h.data)) # ROOT will always return uncert
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

            h.axes[0].label = 'x (cm)'
            h.axes[1].label = 'y (cm)'
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

            h.label = 'counts'
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

            h.title = 'title'
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

        finally:
            os.remove(ftmp.name)

    def test_hist_3d(self):
        h = Histogram(3,[0,1],'xx',4,[-1,1],'yy',5,[5,10],'zz','counts','data')
        ftmp = NamedTemporaryFile(suffix='.root', delete=False)
        try:
            ftmp.close()
            with warnings.catch_warnings(record=True) as w:
                h.save(ftmp.name)
            self.assertEqual(len(w), 1)
            self.assertRegex(str(w[-1].message), 'label')
        finally:
            os.remove(ftmp.name)

    def test_hist_4d(self):
        ftmp = NamedTemporaryFile(suffix='.root', delete=False)
        try:
            h = Histogram(3,[0,1],3,[0,1],3,[0,1],3,[0,1])
            with self.assertRaises(ValueError):
                h.save(ftmp.name)
        finally:
            os.remove(ftmp.name)


if __name__ == '__main__':
    from .. import main
    main()
