from __future__ import division

import unittest

import numpy as np
from numpy.polynomial.polynomial import Polynomial as poly
from numpy import random as rand

from histogram import Histogram

class TestFitting(unittest.TestCase):
    def setUp(self):
        rand.seed(1)

    def test_1dfit(self):
        p = [100,-20,1,-0.2]
        h = Histogram(100,[0,10])
        x = h.axes[0].bincenters
        h.data = poly(p)(x) + rand.normal(0,np.sqrt(p[0]),len(x))

        p0 = [1,1,1,1]
        popt,pcov,ptest = h.fit(lambda x,*p: poly(p)(x), p0)

        assert np.allclose(popt,p,rtol=0.03,atol=0.5)
        assert pcov.shape == (len(p),len(p))

    def test_1dfit_zeros(self):
        p = [100,-20,1,-0.2]
        h = Histogram(100,[0,10])
        x = h.axes[0].bincenters
        h.data = poly(p)(x) + rand.normal(0,np.sqrt(p[0]),len(x))

        ii = rand.randint(0,h.axes[0].nbins,5)
        h.data[ii] = 0

        p0 = [1,1,1,1]
        popt,pcov,ptest = h.fit(lambda x,*p: poly(p)(x), p0)

        assert np.allclose(popt,p,rtol=0.1,atol=1.)
        assert pcov.shape == (len(p),len(p))

    def test_1dfit_nans(self):
        p = [100,-20,1,-0.2]
        h = Histogram(100,[0,10])
        x = h.axes[0].bincenters
        h.data = poly(p)(x) + rand.normal(0,np.sqrt(p[0]),len(x))

        h.data[rand.randint(0,h.axes[0].nbins,5)] = np.nan

        p0 = [1,1,1,1]
        popt,pcov,ptest = h.fit(lambda x,*p: poly(p)(x), p0)

        assert np.allclose(popt,p,rtol=0.03,atol=0.5)
        assert pcov.shape == (len(p),len(p))

    def test_1dfit_infs(self):
        p = [100,-20,1,-0.2]
        h = Histogram(100,[0,10])
        x = h.axes[0].bincenters
        h.data = poly(p)(x) + rand.normal(0,np.sqrt(p[0]),len(x))

        h.data[rand.randint(0,h.axes[0].nbins,5)] = np.inf

        p0 = [1,1,1,1]
        popt,pcov,ptest = h.fit(lambda x,*p: poly(p)(x), p0)

        assert np.allclose(popt,p,rtol=0.03,atol=0.5)
        assert pcov.shape == (len(p),len(p))

    def test_1dfit_nans_infs(self):
        p = [100,-20,1,-0.2]
        h = Histogram(100,[0,10])
        x = h.axes[0].bincenters
        h.data = poly(p)(x) + rand.normal(0,np.sqrt(p[0]),len(x))

        h.data[rand.randint(0,h.axes[0].nbins,5)] = np.nan
        h.data[rand.randint(0,h.axes[0].nbins,5)] = np.inf
        h.data[rand.randint(0,h.axes[0].nbins,5)] = -np.inf

        p0 = [1,1,1,1]
        popt,pcov,ptest = h.fit(lambda x,*p: poly(p)(x), p0)

        assert np.allclose(popt,p,rtol=0.03,atol=0.5)
        assert pcov.shape == (len(p),len(p))

    def test_2dfit(self):
        fn = lambda xy,*p: poly(p[:4])(xy[0]) + poly([0]+list(p[4:]))(xy[1])
        p = [100,-20,1,-0.2,80,-5,0.8]
        h = Histogram(100,[0,10],100,[0,10])
        xy = h.grid
        h.data = fn(xy,*p)
        h.data += rand.normal(0,np.sqrt(p[0]),h.data.shape)

        p0 = [1,1,1,1,1,1,1]
        popt,pcov,ptest = h.fit(fn, p0)

        assert np.allclose(popt,p,rtol=0.01,atol=0.01)
        assert pcov.shape == (len(p),len(p))


if __name__ == '__main__':
    unittest.main()
