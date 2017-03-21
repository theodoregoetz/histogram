# coding: utf-8
import numpy as np
import os
import unittest

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


if __name__ == '__main__':
    from .. import main
    main()
