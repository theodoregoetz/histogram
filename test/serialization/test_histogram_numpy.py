import os
import unittest

from tempfile import NamedTemporaryFile

from histogram import rc as histrc
from histogram import Histogram


class TestSerializationNumpy(unittest.TestCase):
    def setUp(self):
        histrc.overwrite.overwrite = 'always'

    def tearDown(self):
        histrc.overwrite.overwrite = 'ask'

    def test_hist_1d(self):
        ftmp = NamedTemporaryFile(suffix='.hist', delete=False)
        try:
            ftmp.close()
            h = Histogram(3,[0,3])
            h.data[:] = [-3,0,5]
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

            h.axes[0].label = 'x (cm)'
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

            # force underlying numpy storage to hold utf-8 data
            h.axes[0]._label = 'xx'.encode('utf-8')
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            # loaded histogram will have whatever str(''.encode('utf-8'))
            # returns which will be different for py 2/3
            h.axes[0].label = 'xx'.encode('utf-8')
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

    def test_hist_2d(self):
        ftmp = NamedTemporaryFile(suffix='.hist', delete=False)
        try:
            ftmp.close()
            h = Histogram(3,[0,3],4,[0,4])
            h.data[:] = [[-3,0,5,3],[-2,0,4,2],[-1,0,3,1024]]
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

    def test_npz_1d(self):
        ftmp = NamedTemporaryFile(suffix='.npz', delete=False)
        try:
            ftmp.close()
            h = Histogram(3,[0,3])
            h.data[:] = [-3,0,5]
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

    def test_npz_2d(self):
        ftmp = NamedTemporaryFile(suffix='.npz', delete=False)
        try:
            ftmp.close()
            h = Histogram(3,[0,3],4,[0,4])
            h.data[:] = [[-3,0,5,3],[-2,0,4,2],[-1,0,3,1024]]
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
