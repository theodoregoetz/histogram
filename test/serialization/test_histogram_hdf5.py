# coding: utf-8
from __future__ import unicode_literals

import os
import unittest

from tempfile import NamedTemporaryFile

from histogram import rc as histrc
from histogram import Histogram, save_histograms, load_histograms
import histogram


class TestSerializationHDF5(unittest.TestCase):
    def setUp(self):
        histrc.overwrite.overwrite = 'always'

    def tearDown(self):
        histrc.overwrite.overwrite = 'ask'

    def test_unicode(self):
        ftmp = NamedTemporaryFile(suffix='.h5', delete=False)
        try:
            ftmp.close()
            h = Histogram(3,[0,3])
            h.data[:] = [-3,0,5]
            h.title = 'χ-squared'
            h.label = 'αβγ'
            h.axes[0].label = 'θ'
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

        finally:
            os.remove(ftmp.name)

    def test_hist_1d(self):
        ftmp = NamedTemporaryFile(suffix='.h5', delete=False)
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
            #print(type(h.axes[0].label), h.axes[0].label)
            #print(type(htmp.axes[0].label), htmp.axes[0].label)
            self.assertTrue(h.isidentical(htmp))

            h.label = 'counts'
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

            h.title = 'title'
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

            h.uncert = [2,3,4]
            h.save(ftmp.name)
            htmp = Histogram.load(ftmp.name)
            self.assertTrue(h.isidentical(htmp))

        finally:
            os.remove(ftmp.name)

    def test_hist_2d(self):
        ftmp = NamedTemporaryFile(suffix='.hdf5', delete=False)
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

    def test_histograms(self):
        ftmp = NamedTemporaryFile(suffix='.h5', delete=False)
        try:
            ftmp.close()
            hh = dict(
                h0=Histogram(3,[0,1]),
                h1=Histogram(3,[0,1],data=[-1,0,2]),
                h2=Histogram(3,[0,1],data=[-1,0,2],uncert=[0.5,-0.2,1000.]),
                h3=Histogram(3,[0,1],label='counts'),
                h4=Histogram(3,[0,1],label='counts',title='title'),
                h5=Histogram(3,[0,1],4,[-1,1]),
                h6=Histogram(3,[0,1],2,[-1,1],data=[[1,2],[3,4],[5,6]]),
                h7=Histogram(3,[0,1],2,[-1,1],data=[[1,2],[3,4],[5,6]],
                             uncert=[[1,2],[3,4],[5,6]]),
                h8=Histogram(3,[0,1],2,[-1,1],label='counts'),
                h9=Histogram(3,[0,1],2,[-1,1],label='counts',title='τιτλε'))

            save_histograms(hh, ftmp.name)
            hhtmp = load_histograms(ftmp.name)
            #histogram.serialization.serialization.save_histograms(hh, ftmp.name)
            #hhtmp = histogram.serialization.serialization.load_histograms(ftmp.name)

            for k in sorted(hh):
                self.assertTrue(hh[k].isidentical(hhtmp[k]))

        finally:
            os.remove(ftmp.name)


if __name__ == '__main__':
    from .. import main
    main()
