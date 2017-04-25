# coding: utf-8
from __future__ import division

from copy import copy, deepcopy
import numpy as np
import unittest

from numpy.testing import assert_array_almost_equal, assert_array_equal

from histogram import Histogram, HistogramAxis


class TestHistogram(unittest.TestCase):

    def test_init1d(self):
        h1a = Histogram(10,[0,10])
        h1b = Histogram(10,[0,10],'x')
        h1c = Histogram(10,[0,10],'x','label')
        h1d = Histogram(10,[0,10],'x','label','title')

        h2a = Histogram(np.linspace(0,10,11))
        h2b = Histogram(np.linspace(0,10,11),'x')
        h2c = Histogram(np.linspace(0,10,11),'x','label')
        h2d = Histogram(np.linspace(0,10,11),'x','label','title')

        h3a = Histogram((np.linspace(0,10,11),))
        h3b = Histogram((np.linspace(0,10,11),'x'))
        h3c = Histogram((np.linspace(0,10,11),'x'),'label')
        h3d = Histogram((np.linspace(0,10,11),'x'),'label','title')

        self.assertTrue(h1a.isidentical(h2a))
        self.assertTrue(h1b.isidentical(h2b))
        self.assertTrue(h1c.isidentical(h2c))
        self.assertTrue(h1d.isidentical(h2d))

        self.assertTrue(h1a.isidentical(h3a))
        self.assertTrue(h1b.isidentical(h3b))
        self.assertTrue(h1c.isidentical(h3c))
        self.assertTrue(h1d.isidentical(h3d))

    def test_init2d(self):
        h1a = Histogram(10,[0,10],9,[1,10])
        h1b = Histogram(10,[0,10],'x',9,[1,10],'y')
        h1c = Histogram(10,[0,10],'x',9,[1,10],'y','label')
        h1d = Histogram(10,[0,10],'x',9,[1,10],'y','label','title')

        xx = np.linspace(0,10,11)
        yy = np.linspace(1,10,10)
        h2a = Histogram(xx,yy)
        h2b = Histogram(xx,'x',yy,'y')
        h2c = Histogram(xx,'x',yy,'y','label')
        h2d = Histogram(xx,'x',yy,'y','label','title')

        h3a = Histogram((xx,),(yy,))
        h3b = Histogram((xx,'x'),(yy,'y'))
        h3c = Histogram((xx,'x'),(yy,'y'),'label')
        h3d = Histogram((xx,'x'),(yy,'y'),'label','title')

        self.assertTrue(h1a.isidentical(h2a))
        self.assertTrue(h1b.isidentical(h2b))
        self.assertTrue(h1c.isidentical(h2c))
        self.assertTrue(h1d.isidentical(h2d))

        self.assertTrue(h1a.isidentical(h3a))
        self.assertTrue(h1b.isidentical(h3b))
        self.assertTrue(h1c.isidentical(h3c))
        self.assertTrue(h1d.isidentical(h3d))

    def test_init_dtype(self):
        data = np.array([0,1,2], dtype=np.int32)
        h = Histogram(3,[0,3],data=data, dtype=np.int8)
        self.assertEqual(h.data.dtype, np.int8)

    def test_init_failures(self):
        self.assertRaises(TypeError,Histogram)
        with self.assertRaises(TypeError):
            Histogram(2,2)
        if __debug__:
            with self.assertRaises(AssertionError):
                h = Histogram(3,[0,10],data=[0])
            with self.assertRaises(AssertionError):
                h = Histogram(3,[0,10],data=np.array([0], dtype=np.int8))

            with self.assertRaises(ValueError):
                h = Histogram(3,[0,1])
                h.data = np.array([0,1], dtype=np.int8)

        with self.assertRaises(TypeError):
            h = Histogram(2,[0,1],'x','oops',label='label',title='title')

    def test_data(self):
        h = Histogram(3,[0,1])
        h.data = np.array([0,1,2], dtype=np.int8)
        assert_array_almost_equal(h.data, np.array([0,1,2]))

    def test_uncert(self):
        h = Histogram(3, [0,1])
        h.data = [0, 1, 2]
        assert_array_almost_equal(h.uncert, np.sqrt([0, 1, 2]))

        h.uncert = [5,5,5]
        assert_array_almost_equal(h.uncert, [5,5,5])

        h.uncert = None
        assert_array_almost_equal(h.uncert, np.sqrt([0, 1, 2]))

        h.data = [-1, 0, 0]
        del h.uncert
        assert_array_almost_equal(h.uncert, [np.nan, 0, 0])

    def test_uncert_ratio(self):
        h = Histogram(4, [0,1])
        h.data = [-1, 0, 1, 2]
        assert_array_almost_equal(h.uncert_ratio,
                                  [np.nan, np.nan, 1, np.sqrt(2)/2])

        h.data = [-1, 0, 0, 0]
        del h.uncert
        assert_array_almost_equal(h.uncert, [np.nan, 0, 0, 0])

        h.data = [-1, 0, 1, 2]
        h.uncert_ratio = [0.1, 0.1, 0.2, 0.3]
        assert_array_almost_equal(h.uncert, [-0.1, 0, 0.2, 2.*0.3])

    def test_title(self):
        h = Histogram(1,[0,1],title='title')
        self.assertEqual(h.title, 'title')
        h.title = 'test'
        self.assertEqual(h.title, 'test')
        h.title = 'a'
        self.assertEqual(h.title, 'a')
        del h.title
        self.assertIsNone(h.title)

    def test_label(self):
        h = Histogram(1,[0,1],label='label')
        self.assertEqual(h.label, 'label')
        h.label = 'test'
        self.assertEqual(h.label, 'test')
        h.label = 'a'
        self.assertEqual(h.label, 'a')
        del h.label
        self.assertIsNone(h.label)

    def test_eq(self):
        h1 = Histogram(4, (0, 10))
        h2 = Histogram(4, (0, 10))
        h3 = Histogram(4, (-10, 10))
        h4 = Histogram(4, (0, 10))

        h1.data[2] = 1
        h2.data[2] = 1
        h3.data[2] = 1
        h4.data[3] = 1

        self.assertTrue(h1 == h2)
        self.assertFalse(h1 != h2)
        self.assertFalse(h1 == h3)
        self.assertTrue(h1 != h3)
        self.assertFalse(h1 == h4)
        self.assertTrue(h1 != h4)

        h5 = Histogram(5, (0,10))
        self.assertFalse(h1 == h5)

        h6 = Histogram(4, (0, 10))
        h6[2] = 1
        h6.uncert = [0.2]*4
        self.assertTrue(h1 == h6)

        h7 = Histogram(4, (0, 10), label='l')
        h8 = Histogram(4, (0, 10), title='t')
        h7[2] = 1
        h8[2] = 1
        self.assertTrue(h1 == h7)
        self.assertTrue(h1 == h8)

    def test_isidentical(self):
        h1 = Histogram(4, (0, 10))
        h2 = Histogram(4, (0, 10))
        h3 = Histogram(4, (-10, 10))
        h4 = Histogram(4, (0, 10))
        h5 = Histogram(4, (0, 10))
        h6 = Histogram(4, (0, 10), label='l')
        h7 = Histogram(4, (0, 10), title='t')
        h8 = Histogram(4, (0, 10), label='a')
        h9 = Histogram(4, (0, 10), title='b')

        h1.data[2] = 1
        h2.data[2] = 1
        h3.data[2] = 1
        h4.data[3] = 1
        h5.data[2] = 1
        h5.uncert = [0.2]*4
        h6.data[2] = 1
        h7.data[2] = 1
        h8.data[2] = 1
        h9.data[2] = 1

        self.assertTrue(h1.isidentical(h2))
        self.assertFalse(h1.isidentical(h3))
        self.assertFalse(h1.isidentical(h4))
        self.assertFalse(h1.isidentical(h5))
        self.assertFalse(h1.isidentical(h6))
        self.assertFalse(h1.isidentical(h7))
        self.assertFalse(h6.isidentical(h8))
        self.assertFalse(h7.isidentical(h9))

        h1 = Histogram(4, (0,1), 'x', 'h', 't')
        h2 = Histogram(4, (0,1), 'y', 'h', 't')
        self.assertFalse(h1.isidentical(h2))

    def test_str(self):
        h = Histogram(3, (0,1))
        h.data = [1,2,3]
        self.assertEqual(str(h), str(np.array([1,2,3])))

    def test_repr(self):
        h = Histogram(3, (0,3), 'x', 'l', 't')
        h.data = [1,2,3]
        h.uncert = [0,1,2]
        self.assertEqual(repr(h), 'Histogram(HistogramAxis(bins=[0.0, 1.0, 2.0,'
            ' 3.0], label="x"), data=[1, 2, 3], dtype="int64", label="l",'
            ' title="t", uncert=[0.0, 1.0, 2.0])')

        h = Histogram(3, (0,3))
        self.assertEqual(repr(h), 'Histogram(HistogramAxis(bins=[0.0, 1.0, 2.0,'
            ' 3.0]), data=[0, 0, 0], dtype="int64")')

    def test_call(self):
        h1 = Histogram(10,[0,10],data=[x+10 for x in range(10)])
        h2 = Histogram(10,[0,10],9,[-10,-1])
        h2.data[3,3] = 5
        self.assertAlmostEqual(h1(5),15)
        self.assertAlmostEqual(h1(20, overflow_value=50), 50)

    def test_asdict(self):
        h = Histogram(3,[0,3],'xx','ll','tt',data=[5,6,7])
        d = h.asdict()
        assert_array_almost_equal(d['data'], [5,6,7])
        for a,aa in zip(h.axes, d['axes']):
            self.assertEqual(a, HistogramAxis.fromdict(aa))
        self.assertEqual(d['label'], 'll')
        self.assertEqual(d['title'], 'tt')
        self.assertNotIn('uncert', d)
        self.assertTrue(h.isidentical(Histogram.fromdict(d)))
        h.uncert = [3,4,5]
        d = h.asdict()
        assert_array_almost_equal(d['uncert'], [3,4,5])
        self.assertTrue(h.isidentical(Histogram.fromdict(d)))

        h = Histogram(3,[0,3])
        d = h.asdict()
        self.assertTrue(h.isidentical(Histogram.fromdict(d)))

    def test_asdict_flat(self):
        h = Histogram(3,[0,3],'xx','ll','tt',data=[5,6,7])
        d = h.asdict(flat=True)
        assert_array_almost_equal(d['data'], [5,6,7])
        self.assertEqual(d['label'], 'll')
        self.assertEqual(d['title'], 'tt')
        assert_array_almost_equal(d['axes:0:edges'], [0,1,2,3])
        self.assertEqual(d['axes:0:label'], 'xx')
        self.assertTrue(h.isidentical(Histogram.fromdict(d)))
        h.uncert = [8,9,10]
        d = h.asdict(flat=True)
        assert_array_almost_equal(d['uncert'], [8,9,10])
        self.assertTrue(h.isidentical(Histogram.fromdict(d)))

    def test_asdict_encoding(self):
        h = Histogram(3,[0,3],'xx','ll','tt',data=[5,6,7])
        d = h.asdict(encoding='ascii')
        self.assertTrue(h.isidentical(Histogram.fromdict(d, 'ascii')))
        d = h.asdict(encoding='utf-8')
        self.assertTrue(h.isidentical(Histogram.fromdict(d, 'utf-8')))

    def test_dim(self):
        h1 = Histogram(3,[0,1])
        h2 = Histogram(3,[0,1],4,[0,1])
        h3 = Histogram(3,[0,1],4,[0,1],5,[0,1])
        self.assertEqual(h1.dim, 1)
        self.assertEqual(h2.dim, 2)
        self.assertEqual(h3.dim, 3)

    def test_shape(self):
        h1 = Histogram(3,[0,1])
        h2 = Histogram(3,[0,1],4,[0,1])
        h3 = Histogram(3,[0,1],4,[0,1],5,[0,1])
        self.assertEqual(h1.shape, (3,))
        self.assertEqual(h2.shape, (3,4))
        self.assertEqual(h3.shape, (3,4,5))

    def test_size(self):
        h1 = Histogram(3,[0,1])
        h2 = Histogram(3,[0,1],4,[0,1])
        h3 = Histogram(3,[0,1],4,[0,1],5,[0,1])
        self.assertEqual(h1.size, 3)
        self.assertEqual(h2.size, 3*4)
        self.assertEqual(h3.size, 3*4*5)

    def test_isuniform(self):
        h = Histogram([0,1,2])
        self.assertTrue(h.isuniform())
        h = Histogram([0,1,2], [-1,0,1])
        self.assertTrue(h.isuniform())
        h = Histogram([0,1,3])
        self.assertFalse(h.isuniform())
        h = Histogram([0,1,2], [-1,0,2])
        self.assertFalse(h.isuniform())

    def test_edges(self):
        h = Histogram([0,1,2])
        self.assertEqual(len(h.edges), 1)
        assert_array_almost_equal(h.edges[0], [0,1,2])
        h = Histogram([0,1,2], [-1,0,1])
        self.assertEqual(len(h.edges), 2)
        assert_array_almost_equal(h.edges[0], [0,1,2])
        assert_array_almost_equal(h.edges[1], [-1,0,1])

    def test_grid(self):
        h = Histogram([0,1,2])
        self.assertEqual(len(h.grid()), 1)
        assert_array_almost_equal(h.grid()[0], [0.5,1.5])
        h = Histogram([0,1,2], [-1,0,1])
        self.assertEqual(len(h.grid()), 2)
        assert_array_almost_equal(len(h.grid()[0]), 2)
        assert_array_almost_equal(len(h.grid()[1]), 2)
        assert_array_almost_equal(h.grid()[0].ravel(), [0.5,0.5,1.5,1.5])
        assert_array_almost_equal(h.grid()[1].ravel(), [-0.5,0.5,-0.5,0.5])

    def test_edge_grid(self):
        h = Histogram([0,1,2])
        self.assertEqual(len(h.edge_grid()), 1)
        assert_array_almost_equal(h.edge_grid()[0], [0,1,2])
        h = Histogram([0,1,2], [-1,0,1])
        self.assertEqual(len(h.edge_grid()), 2)
        assert_array_almost_equal(len(h.edge_grid()[0]), 3)
        assert_array_almost_equal(len(h.edge_grid()[1]), 3)
        assert_array_almost_equal(h.edge_grid()[0].ravel(), [0,0,0,1,1,1,2,2,2])
        assert_array_almost_equal(h.edge_grid()[1].ravel(), [-1,0,1]*3)

    def test_binwidths(self):
        h = Histogram([0,1,2])
        self.assertEqual(len(h.binwidths()), 1)
        assert_array_almost_equal(h.binwidths()[0], [1,1])
        h = Histogram([0,2,4], [-1,0,2])
        self.assertEqual(len(h.binwidths()), 2)
        assert_array_almost_equal(h.binwidths()[0], [2,2])
        assert_array_almost_equal(h.binwidths()[1], [1,2])

    def test_binwidth(self):
        h = Histogram([0,1,2])
        self.assertEqual(h.binwidth(0), 1)
        self.assertEqual(h.binwidth(1), 1)

        h = Histogram([0,2,4], [-1,0,2])
        self.assertEqual(h.binwidth(0), 2)
        self.assertEqual(h.binwidth(1), 2)
        self.assertEqual(h.binwidth(0,1), 1)
        self.assertEqual(h.binwidth(1,1), 2)

    def test_binvolumes(self):
        h = Histogram([0,1,3])
        self.assertEqual(tuple(len(x) for x in h.binvolumes()), (2,))
        self.assertEqual(h.binvolumes()[0][0], 1)
        self.assertEqual(h.binvolumes()[0][1], 2)

        h = Histogram([0,2,4], [-1,0,2,6])
        self.assertEqual(len(h.binvolumes()), 2)
        self.assertEqual(len(h.binvolumes()[0]), 3)
        self.assertEqual(len(h.binvolumes()[1]), 3)
        self.assertEqual(h.binvolumes()[0,0], 2)
        self.assertEqual(h.binvolumes()[0,1], 4)
        self.assertEqual(h.binvolumes()[0,2], 8)
        self.assertEqual(h.binvolumes()[1,0], 2)
        self.assertEqual(h.binvolumes()[1,1], 4)
        self.assertEqual(h.binvolumes()[1,2], 8)

    def test_overflow_value(self):
        h = Histogram([0,1,2])
        self.assertEqual(h.overflow_value, (3,))

        h = Histogram([0,2,4], [-1,0,2])
        self.assertEqual(h.overflow_value, (5,3))

    def test_sum1d(self):
        h = Histogram(10, [0, 10])
        h.fill([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        self.assertEqual(h.sum().n, 10)
        self.assertEqual(h.sum().s, np.sqrt(10))
        h.fill(11)
        self.assertEqual(h.sum().n, 10)
        self.assertEqual(h.sum(0).n, 10)

        h.uncert = h.uncert
        s = h.sum()
        self.assertAlmostEqual(s.n, 10)
        self.assertAlmostEqual(s.s, 3.16227766)

    def test_sum2d(self):
        h2 = Histogram(10, [0, 10], 10, [0, 10])
        h2.fill([1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
                [1, 1, 1, 1, 1, 1, 2, 2, 2, 2])

        h2x_exp = Histogram(10, [0,10])
        h2x_exp.fill([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        h2y_exp = Histogram(10, [0,10])
        h2y_exp.fill([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])

        self.assertEqual(h2.sum().n, 10)
        h2y = h2.sum(0)
        h2x = h2.sum(1)
        self.assertTrue(h2y.isidentical(h2y_exp))
        self.assertTrue(h2x.isidentical(h2x_exp))

        h2.uncert = h2.uncert
        hx = h2.sum(1)
        hy = h2.sum(0)
        self.assertAlmostEqual(hx.sum().n, hy.sum().n)
        self.assertEqual(hx.shape, (10,))
        self.assertEqual(hy.shape, (10,))

        assert_array_almost_equal(hx.data,   [0,1,2,3,4,0,0,0,0,0])
        assert_array_almost_equal(hx.uncert, np.sqrt([0,1,2,3,4,0,0,0,0,0]))
        assert_array_almost_equal(hy.data,   [0,6,4,0,0,0,0,0,0,0])
        assert_array_almost_equal(hy.uncert, np.sqrt([0,6,4,0,0,0,0,0,0,0]))

    def test_sum3d(self):
        xx = [1,1,5,5,5,9,9,9,9]
        yy = [1,1,1,1,9,9,9,9,9]
        zz = [5,5,5,5,5,9,9,9,9]
        h2 = Histogram(3, [0, 10], 4, [0, 10], 5, [0, 10])
        h2.fill(xx, yy, zz)

        self.assertEqual(h2.sum().n, 9)

        h2x_exp = Histogram(3, [0,10])
        h2x_exp.fill(xx)
        h2y_exp = Histogram(4, [0,10])
        h2y_exp.fill(yy)
        h2z_exp = Histogram(5, [0,10])
        h2z_exp.fill(zz)

        h2xy_exp = Histogram(3, [0,10], 4, [0,10])
        h2xy_exp.fill(xx,yy)
        h2xz_exp = Histogram(3, [0,10], 5, [0,10])
        h2xz_exp.fill(xx,zz)
        h2yz_exp = Histogram(4, [0,10], 5, [0,10])
        h2yz_exp.fill(yy,zz)

        self.assertTrue(h2.sum(1,2).isidentical(h2x_exp))
        self.assertTrue(h2.sum(0,2).isidentical(h2y_exp))
        self.assertTrue(h2.sum(0,1).isidentical(h2z_exp))

        self.assertTrue(h2.sum(2).isidentical(h2xy_exp))
        self.assertTrue(h2.sum(1).isidentical(h2xz_exp))
        self.assertTrue(h2.sum(0).isidentical(h2yz_exp))

    def test_sum_nonfinite(self):
        h = Histogram(3,[0,3], dtype=np.float)
        h.data = [np.nan, -1, 2]
        self.assertTrue(np.isnan(h.sum().n))

        h.data = [np.inf, -1, 2]
        self.assertEqual(h.sum().n, np.inf)

        h.data = [-np.inf, -1, 2]
        self.assertEqual(h.sum().n, -np.inf)

        h.data = [np.inf, np.nan, 2]
        self.assertTrue(np.isnan(h.sum().n))

    def test_projection(self):
        h2 = Histogram(2, [0,1], 3, [0,9], data=[[-1,2,3],[4,2,-4]])

        hx = h2.projection(0)
        hy = h2.projection(1)

        assert_array_almost_equal(hx.data, [4,2])
        assert_array_almost_equal(hy.data, [3,4,-1])
        self.assertFalse(hx.has_uncert)
        self.assertFalse(hy.has_uncert)

    def test_projection_uncert(self):
        h2 = Histogram(2, [0,1], 3, [0,9], data=[[-1,2,3],[4,2,-4]])
        h2.uncert = [[.1,.2,.3],[.4,.5,.6]]

        hx = h2.projection(0)
        hy = h2.projection(1)

        assert_array_almost_equal(hx.data, [4,2])
        assert_array_almost_equal(hy.data, [3,4,-1])

        xuncert = [
            np.sqrt(np.sum(np.array([.1,.2,.3])**2)),
            np.sqrt(np.sum(np.array([.4,.5,.6])**2))]
        yuncert = [
            np.sqrt(np.sum(np.array([.1,.4])**2)),
            np.sqrt(np.sum(np.array([.2,.5])**2)),
            np.sqrt(np.sum(np.array([.3,.6])**2))]

        assert_array_almost_equal(hx.uncert, xuncert)
        assert_array_almost_equal(hy.uncert, yuncert)

    def test_integral(self):
        h1 = Histogram(4, [0,8], data=[2,3,4,5])
        h2 = Histogram(2, [0,1], 3, [0,9], data=[[-1,2,3],[4,2,-4]])
        h2.uncert = np.sqrt(np.abs(h2.data))

        def calc_integral(h, use_uncert=True):
            bv = h.binvolumes()
            res = np.sum(h.data * bv)
            if use_uncert:
                e = np.sqrt(np.sum(h.uncert * h.uncert * bv * bv))
            else:
                e = np.sqrt(np.sum(h.data * bv * bv))
            return res, e

        assert_array_almost_equal(h1.integral(), calc_integral(h1))
        assert_array_almost_equal(h1.integral(), calc_integral(h1, False))
        assert_array_almost_equal(h2.integral(), calc_integral(h2))

    def test_minmax(self):
        h1 = Histogram(4, [0,8], data=[2,3,4,5])
        h2 = Histogram(2, [0,1], 3, [0,9], data=[[-1,2,3],[5,2,-4]])

        self.assertAlmostEqual(h1.min(), 2 - np.sqrt(2))
        self.assertAlmostEqual(h1.max(), 5 + np.sqrt(5))

        self.assertAlmostEqual(h2.min(), 2 - np.sqrt(2))
        self.assertAlmostEqual(h2.max(), 5 + np.sqrt(5))

        h2.uncert = .2
        self.assertAlmostEqual(h2.min(), -4 - .2)
        self.assertAlmostEqual(h2.max(), 5 + .2)

    def test_mean_1d(self):
        h = Histogram(10,[0,10])
        h.fill([3,3,3])
        self.assertEqual(len(h.mean()), 1)
        m = h.mean()[0]
        self.assertAlmostEqual(m.n, 3.5)
        self.assertAlmostEqual(m.s, 0.5)

        h.fill([1,5])
        m = h.mean()[0]
        self.assertAlmostEqual(m.n, 3.5)
        self.assertAlmostEqual(m.s, 0.65574385243)

        h.fill([7,7,7,8,9])
        m = h.mean()[0]
        self.assertAlmostEqual(m.n, 5.8)
        self.assertAlmostEqual(m.s, 0.83426614458)

        h = Histogram(5, [100-12.5, 200+12.5])
        h.data[:] = [10, 12, 20, 8, 5]
        self.assertAlmostEqual(h.mean()[0].n, 143+(14/22))

    def test_mean_2d(self):
        h = Histogram(10,[0,10],15,[0,20])
        h.fill([3,3,3,5,5,5], [7,7,7,9,9,9])
        m = h.mean()
        self.assertEqual(len(m), 2)
        self.assertAlmostEqual(m[0].n, 4.5)
        self.assertAlmostEqual(m[0].s, 0.540061724867)
        self.assertAlmostEqual(m[1].n, 8.0)
        self.assertAlmostEqual(m[1].s, 0.54433105395)

    def test_mean_nonuniform(self):
        h = Histogram([0,2,3,4])
        h.data[:] = [3,2,1]
        m = h.mean()
        self.assertAlmostEqual(m[0].n, 1.61111111111111)
        self.assertAlmostEqual(m[0].s, 0.76076334009)

    def test_var_1d(self):
        h = Histogram(10,[0,10])
        h.fill([3,3,3])
        self.assertEqual(len(h.var()), 1)
        v = h.var()[0]
        self.assertAlmostEqual(v.n, 0.0)
        self.assertAlmostEqual(v.s, 0.0)

        h.fill([1,5])
        v = h.var()[0]
        self.assertAlmostEqual(v.n, 1.6)
        self.assertAlmostEqual(v.s, 1.0430723848)

        h.fill([7,7,7,8,9])
        v = h.var()[0]
        self.assertAlmostEqual(v.n, 6.41)
        self.assertAlmostEqual(v.s, 1.9843286018)

    def test_var_2d(self):
        h = Histogram(10,[0,10],15,[0,20])
        h.fill([3,3,3,5,5,5], [7,7,7,9,9,9])
        v = h.var()
        self.assertEqual(len(v), 2)
        self.assertAlmostEqual(v[0].n, 1.0)
        self.assertAlmostEqual(v[0].s, 0.7071067811865)
        self.assertAlmostEqual(v[1].n, 0.4444444444444)
        self.assertAlmostEqual(v[1].s, 0.628539361055)

    def test_var_nonuniform(self):
        h = Histogram([0,2,3,4])
        h.data[:] = [3,2,1]
        v = h.var()
        self.assertAlmostEqual(v[0].n, 0.82098765432)
        self.assertAlmostEqual(v[0].s, 0.932651991698)

    def test_std_1d(self):
        h = Histogram(10,[0,10])
        h.fill([3,3,3])
        self.assertEqual(len(h.std()), 1)
        v = h.std()[0]
        self.assertAlmostEqual(v.n, 0.0)
        self.assertAlmostEqual(v.s, 0.0)

        h.fill([1,5])
        v = h.std()[0]
        self.assertAlmostEqual(v.n, np.sqrt(1.6))
        self.assertAlmostEqual(v.s, 0.41231056256)

        h.fill([7,7,7,8,9])
        v = h.std()[0]
        self.assertAlmostEqual(v.n, np.sqrt(6.41))
        self.assertAlmostEqual(v.s, 0.3918813377)

    def test_std_2d(self):
        h = Histogram(10,[0,10],15,[0,20])
        h.fill([3,3,3,5,5,5], [7,7,7,9,9,9])
        v = h.std()
        self.assertEqual(len(v), 2)
        self.assertAlmostEqual(v[0].n, 1.0)
        self.assertAlmostEqual(v[0].s, 0.35355339)
        self.assertAlmostEqual(v[1].n, np.sqrt(0.4444444444444))
        self.assertAlmostEqual(v[1].s, 0.4714045208)

    def test_extent_1d(self):
        h = Histogram(10,[0,10])
        assert_array_almost_equal(h.extent(), [0,10,0,0])
        assert_array_almost_equal(h.extent(uncert=False), [0,10,0,0])
        assert_array_almost_equal(h.extent(pad=1), [-10,20,0,0])

        h.fill([3,3,3])
        ex = [0,10,0,3+np.sqrt(3)]
        assert_array_almost_equal(h.extent(), ex)
        assert_array_almost_equal(h.extent(uncert=False), [0,10,0,3])
        assert_array_almost_equal(h.extent(uncert=False,pad=1), [-10,20,-3,6])

        ex = [-10,20,0,3+np.sqrt(3)]
        w = 3 + np.sqrt(3)
        ex[2] -= w
        ex[3] += w
        assert_array_almost_equal(h.extent(pad=1), ex)

        ex = [-10,30,-3*w,5*w]
        assert_array_almost_equal(h.extent(pad=[1,2,3,4]), ex)

        assert_array_almost_equal(h.extent(1), [0,10])

    def test_extent_2d(self):
        h = Histogram(10,[0,10],15,[0,20])
        assert_array_almost_equal(h.extent(2), [0,10,0,20])

        h.fill([3,3,3,5,5,5], [7,7,7,9,9,9])

        ex = [0,10,0,20,0,3]
        assert_array_almost_equal(h.extent(uncert=False), ex)
        ex[-1] = 3 + np.sqrt(3)
        assert_array_almost_equal(h.extent(), ex)

    def test_errorbars(self):
        h = Histogram([0,2,3,4])
        h.data[:] = [1,2,5]
        h.uncert = [0,1,2]
        ebars = h.errorbars()
        assert_array_almost_equal(ebars[0], [1,0.5,0.5])
        assert_array_almost_equal(ebars[1], [0,1,2])
        ebars = h.errorbars(1)
        assert_array_almost_equal(ebars[0], [1,0.5,0.5])
        ebars = h.errorbars(asratio=True)
        assert_array_almost_equal(ebars[0], [.25, .125, .125])
        assert_array_almost_equal(ebars[1], [0, .25, .5])
        ebars = h.errorbars(1,asratio=True)
        assert_array_almost_equal(ebars[0], [.25, .125, .125])
        self.assertEqual(len(ebars), 1)

    def test_asline(self):
        h = Histogram(10,[0,10])
        h.data = np.array([1,2,3,4,5,0,1,2,9,0],dtype=np.int64)
        x,y,ext = h.asline()
        assert_array_almost_equal(x,[0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10])
        assert_array_almost_equal(y,[1,1,2,2,3,3,4,4,5,5,0,0,1,1,2,2,9,9,0, 0])
        assert_array_almost_equal(ext,[0,10,0,9])

        h = Histogram(10,[0,10])
        h.data = np.array([0,2,3,4,5,0,1,2,9,0],dtype=np.int64)
        x,y,ext = h.asline()
        assert_array_almost_equal(x,[0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10])
        assert_array_almost_equal(y,[0,0,2,2,3,3,4,4,5,5,0,0,1,1,2,2,9,9,0, 0])
        assert_array_almost_equal(ext,[0,10,0,9])

        h = Histogram(10,[0,10])
        h.data = np.array([0,2,-4,4,5,0,1,2,9,0],dtype=np.int64)
        x,y,ext = h.asline()
        assert_array_almost_equal(x,[0,1,1,2, 2, 3,3,4,4,5,5,6,6,7,7,8,8,9,9,10])
        assert_array_almost_equal(y,[0,0,2,2,-4,-4,4,4,5,5,0,0,1,1,2,2,9,9,0, 0])
        assert_array_almost_equal(ext,[0,10,-4,9])

        h = Histogram(10,[-1,9])
        h.data = np.array([-1,2,-4,4,5,0,1,2,9,-10],dtype=np.int64)
        x,y,ext = h.asline()
        assert_array_almost_equal(x,[-1, 0,0,1, 1, 2,2,3,3,4,4,5,5,6,6,7,7,8,  8,  9])
        assert_array_almost_equal(y,[-1,-1,2,2,-4,-4,4,4,5,5,0,0,1,1,2,2,9,9,-10,-10])
        assert_array_almost_equal(ext,[-1,9,-10,9])

    def test_getitem_setitem(self):
        h = Histogram(10,[0,10])
        h[:] = list(range(10))
        assert_array_almost_equal(h[:], list(range(10)))

        h = Histogram(3,[0,10],4,[0,10])
        h[...] = np.ones(shape=(3,4))
        assert_array_almost_equal(h[:,0], [1,1,1])
        assert_array_almost_equal(h[0,:], [1,1,1,1])

    def test_set(self):
        h = Histogram(3,[0,10])
        h.set(3)
        assert_array_almost_equal(h.data, [3,3,3])

        h.uncert = [5,5,5]
        self.assertTrue(h.has_uncert)
        h.set(3)
        self.assertFalse(h.has_uncert)
        h.set(3, 6)
        assert_array_almost_equal(h.uncert, [6,6,6])

        h.set(list(range(3)), 6)
        assert_array_almost_equal(h.data, [0,1,2])
        assert_array_almost_equal(h.uncert, [6,6,6])

        h = Histogram(3,[0,10],4,[0,10])
        h.set(np.ones(shape=(3,4)), np.full((3,4), 5, dtype=np.float64))
        assert_array_almost_equal(h[:,0], [1,1,1])
        assert_array_almost_equal(h[0,:], [1,1,1,1])
        assert_array_almost_equal(h.uncert[:,0], [5,5,5])

    def test_set_nans(self):
        h = Histogram(3,[0,10],dtype=np.float)

        h[:] = [1,np.nan,np.inf]
        h.set_nans(0)
        assert_array_almost_equal(h.data, [1,0,np.inf])
        self.assertFalse(h.has_uncert)

        h[:] = [1,np.nan,np.inf]
        h.uncert = [1,np.nan,np.inf]
        h.set_nans(0,1)
        assert_array_almost_equal(h.data, [1,0,np.inf])
        assert_array_almost_equal(h.uncert, [1,1,np.inf])

    def test_set_infs(self):
        h = Histogram(3,[0,10],dtype=np.float)

        h[:] = [1,np.nan,np.inf]
        h.set_infs(0)
        assert_array_almost_equal(h.data, [1,np.nan,0])
        self.assertFalse(h.has_uncert)

        h[:] = [1,np.nan,np.inf]
        h.uncert = [1,np.nan,np.inf]
        h.set_infs(0,1)
        assert_array_almost_equal(h.data, [1,np.nan,0])
        assert_array_almost_equal(h.uncert, [1,np.nan,1])

    def test_set_nonfinites(self):
        h = Histogram(3,[0,10],dtype=np.float)

        h[:] = [1,np.nan,np.inf]
        h.set_nonfinites(0,1)
        assert_array_almost_equal(h.data, [1,0,0])
        self.assertFalse(h.has_uncert)

        h[:] = [1,np.nan,np.inf]
        h.uncert = [1,np.nan,np.inf]
        h.set_nonfinites(0,1)
        assert_array_almost_equal(h.data, [1,0,0])
        assert_array_almost_equal(h.uncert, [1,1,1])

    def test_reset(self):
        h = Histogram(3,[0,10],dtype=np.float)
        h[:] = [1,np.nan,np.inf]
        h.uncert = [1,np.nan,np.inf]
        h.reset()
        assert_array_almost_equal(h.data, [0,0,0])
        self.assertIsNone(getattr(h, '_uncert', None))

    def test_fill_1d(self):
        h = Histogram(10, [0, 10])

        h.fill(1)
        h.fill(2, 2)
        h.fill([3, 3, 3])
        h.fill([4, 4], 2)
        h.fill([5, 5], [2, 3])

        assert_array_almost_equal(h.data, [0, 1, 2, 3, 4, 5, 0, 0, 0, 0])

    def test_fill_2d(self):
        h = Histogram(3, [0, 3], 10, [0, 10])
        xdata = [0, 0, 1, 1, 2]
        ydata = [1, 2, 3, 4, 5]
        weights = [1, 2, 3, 4, 5]
        h.fill(xdata, ydata, weights)

        assert_array_almost_equal(h.data,
            np.array([[0,1,2,0,0,0,0,0,0,0],
                      [0,0,0,3,4,0,0,0,0,0],
                      [0,0,0,0,0,5,0,0,0,0]]))

    def test_fill_one_1d(self):
        h = Histogram(10, [0, 10])
        h.fill_one(1)
        h.fill_one(2,2)
        for x in [3,3,3]:
            h.fill_one(x)
        for x in [4,4]:
            h.fill_one(x,2)
        for x,w in zip([5,5],[2,3]):
            h.fill_one(x,w)
        h.fill_one(100)
        assert_array_almost_equal(h.data, [0, 1, 2, 3, 4, 5, 0, 0, 0, 0])

    def test_fill_one_2d(self):
        h = Histogram(3, [0, 3], 10, [0, 10])
        for x,y,w in zip([0, 0, 1, 1, 2],
                         [1, 2, 3, 4, 5],
                         [1, 2, 3, 4, 5]):
            h.fill_one((x,y),w)
        h.fill_one((2.9,9.9))
        h.fill_one((100,5))
        h.fill_one((2,100))

        assert_array_almost_equal(h.data,
            np.array([[0,1,2,0,0,0,0,0,0,0],
                      [0,0,0,3,4,0,0,0,0,0],
                      [0,0,0,0,0,5,0,0,0,1]]))

    def test_fill_from_sample(self):
        h = Histogram(3, [0, 3], 10, [0, 10])
        xdata = [0, 0, 1, 1, 2]
        ydata = [1, 2, 3, 4, 5]
        weights = [1, 2, 3, 4, 5]
        h.fill_from_sample((xdata, ydata), weights)

        assert_array_almost_equal(h.data,
            np.array([[0,1,2,0,0,0,0,0,0,0],
                      [0,0,0,3,4,0,0,0,0,0],
                      [0,0,0,0,0,5,0,0,0,0]]))

    def test_copy(self):
        h = Histogram(3, [0, 3], 10, [0, 10])
        xdata = [0, 0, 1, 1, 2]
        ydata = [1, 2, 3, 4, 5]
        weights = [1, 2, 3, 4, 5]
        h.fill(xdata, ydata, weights)

        h1 = deepcopy(h)
        h.isidentical(h1)
        h1 = copy(h)
        h.isidentical(h1)
        h1 = h.copy()
        h.isidentical(h1)

    def test_add(self):
        h1 = Histogram(3,[0,10],data=[1,2,3])

        h = h1 + 1
        self.assertEqual(h.data.dtype, np.dtype('int64'))
        assert_array_almost_equal(h1.data, [1,2,3])
        assert_array_almost_equal(h.data,  [2,3,4])
        assert_array_almost_equal(h.uncert,  np.sqrt([1,2,3]))

        h = h1 + [2,3,4]
        self.assertEqual(h.data.dtype, np.dtype('float64'))
        assert_array_almost_equal(h1.data, [1,2,3])
        assert_array_almost_equal(h.data,  [3,5,7])
        assert_array_almost_equal(h.uncert,  np.sqrt([1,2,3]))

        h = h1 + np.array([2,3,4])
        self.assertEqual(h.data.dtype, np.dtype('float64'))
        assert_array_almost_equal(h1.data, [1,2,3])
        assert_array_almost_equal(h.data,  [3,5,7])
        assert_array_almost_equal(h.uncert,  np.sqrt([1,2,3]))

        h2 = Histogram(3,[0,10],data=[4,5,6])

        h = h1 + h2
        self.assertEqual(h.data.dtype, np.dtype('int64'))
        assert_array_almost_equal(h1.data, [1,2,3])
        assert_array_almost_equal(h.data,  [5,7,9])
        assert_array_almost_equal(h.uncert,  np.sqrt([5,7,9]))

        h1 = Histogram(3,[0,1],dtype=np.int64)
        h2 = Histogram(3,[0,1],dtype=np.int32)
        h = h1 + h2
        self.assertEqual(h.data.dtype, np.dtype('int64'))

    def test_radd(self):
        h1 = Histogram(3,[0,1],data=[1,2,3])
        h = 1 + h1
        self.assertEqual(h.data.dtype, np.int64)
        assert_array_equal(h.data, [2,3,4])

    def test_add_uncert(self):
        h1 = Histogram(5,[0,10],data=[-2,-1,0,1,2])
        h2 = Histogram(5,[0,10],data=[2,3,4,5,6])

        h3 = h1 + h2
        self.assertEqual(h3.data.dtype, np.dtype('int64'))
        assert_array_almost_equal(h3.data,  [0,2,4,6,8])
        assert_array_almost_equal(h3.uncert,  np.sqrt([0,2,4,6,8]))

        h1.uncert = h1.uncert
        h3 = h1 + h2
        self.assertEqual(h3.data.dtype, np.dtype('float64'))
        assert_array_almost_equal(h3.data,  [0,2,4,6,8])
        assert_array_almost_equal(h3.uncert,
                                  [np.nan,np.nan,2,2.44948974,2.82842712])

    def test_add_broadcast(self):
        h1 = Histogram(2,[0,1],3,[0,1],data=[[0,1,2],[3,4,5]])
        h2 = Histogram(2,[0,1],data=[3,4])
        h3 = h1 + h2
        self.assertEqual(h3.shape, h1.shape)
        assert_array_almost_equal(h3.data, [[3,4,5],[7,8,9]])

        h3 = h2 + h1
        self.assertEqual(h3.shape, h1.shape)
        assert_array_almost_equal(h3.data, [[3,4,5],[7,8,9]])

    def test_iadd(self):
        h1 = Histogram(3,[0,10],data=[1,2,3])

        h1 += 1
        assert_array_almost_equal(h1.data, [2,3,4])

        h1 += [2,3,4]
        assert_array_almost_equal(h1.data, [4,6,8])

        h2 = Histogram(3,[0,10],data=[4,5,6])

        h1 += h2
        assert_array_almost_equal(h1.data, [8,11,14])

    def test_sub(self):
        h1 = Histogram(3,[0,10],data=[10,10,10])
        h2 = Histogram(3,[0,10],data=[ 4, 5, 6])
        h3 = h1 - h2
        assert_array_almost_equal(h3.data, [6,5,4])

    def test_isub(self):
        h1 = Histogram(3,[0,10],data=[1,2,3])

        h1 -= 1
        assert_array_almost_equal(h1.data, [0,1,2])

        h1 -= [2,3,4]
        assert_array_almost_equal(h1.data, [-2,-2,-2])

        h1 = Histogram(3,[0,10],data=[10,10,10])
        h2 = Histogram(3,[0,10],data=[ 4, 5, 6])
        h1 -= h2
        assert_array_almost_equal(h1.data, [6,5,4])

    def test_rsub(self):
        h1 = Histogram(3,[0,10],data=[10,10,10])
        h3 = [11,12,13] - h1
        assert_array_almost_equal(h3.data, [1,2,3])

    def test_imul(self):
        h1 = Histogram(3,[0,10],data=[1,2,3])
        h1 *= 2.
        self.assertEqual(h1.data.dtype,np.int64)

        h2 = Histogram(3,[0,10],data=[2,2,3])
        h1 *= h2
        assert_array_almost_equal(h1.data, [4, 8, 18])

    def test_mul(self):
        h1 = Histogram(3,[0,10],data=[1,2,3])
        h2 = h1 * 2.
        self.assertEqual(h2.data.dtype,np.float)

        h2 = Histogram(3,[0,10],data=[2,2,3])
        h3 = h1 * h2
        assert_array_almost_equal(h3.data, [2,4,9])

    def test_mul_uncert(self):
        h1 = Histogram(3,[0,10],data=[1,2,3],uncert=[1,2,3])
        h2 = h1 * 2
        assert_array_almost_equal(h2.uncert,[2,4,6])
        h3 = h1 * h2
        uncrat = np.sqrt((h1.uncert/h1.data)**2 + (h2.uncert/h2.data)**2)
        assert_array_almost_equal(h3.uncert,uncrat * h3.data)

    def test___rmul__(self):
        h1 = Histogram(3,[0,10],data=[1,2,3])
        h2 = 2. * h1
        self.assertEqual(h2.data.dtype,np.float)

        h1 = Histogram(3,[0,10],data=[10,10,10])
        h3 = [11,12,13] * h1
        assert_array_almost_equal(h3.data, [110,120,130])

    def test_truediv(self):
        h1 = Histogram(3,[0,10],data=[1,2,3])
        h2 = Histogram(3,[0,10],data=[2,1,0])

        h3 = h1 / 2
        assert_array_equal(h1.data, np.array([1,2,3],dtype=np.int64))
        assert_array_almost_equal(h3.data,  [0.5,1,1.5])

        h3 = h2 / h1
        assert_array_equal(h1.data, np.array([1,2,3],dtype=np.int64))
        assert_array_equal(h2.data, np.array([2,1,0],dtype=np.int64))
        assert_array_almost_equal(h3.data,  [2.0,0.5,0.0])

        h3 = h1 / h2
        assert_array_equal(h1.data, np.array([1,2,3],dtype=np.int64))
        assert_array_equal(h2.data, np.array([2,1,0],dtype=np.int64))
        assert_array_almost_equal(h3.data, [0.5,2.0,np.inf])

        h1.data[:] = [1,-2,3]
        h2.data[:] = [1,0,0]
        h3 = h1 / h2
        assert_array_almost_equal(h3.data, [1,-np.inf,np.inf])

    def test_div_uncert(self):
        h1 = Histogram(3,[0,10],data=[1,2,3],uncert=[1,2,3])
        h2 = h1 / 2
        assert_array_almost_equal(h2.uncert,[0.5,1,1.5])
        h3 = h1 / h2
        uncrat = np.sqrt((h1.uncert/h1.data)**2 + (h2.uncert/h2.data)**2)
        assert_array_almost_equal(h3.uncert,uncrat * h3.data)

    def test_itruediv(self):
        h1 = Histogram(3,[0,10],data=[1,2,3])
        h2 = Histogram(3,[0,10],data=[2,1,0])

        h3 = h1.copy(np.float64)
        h3 /= 2
        assert_array_equal(h1.data, np.array([1,2,3],dtype=np.int64))
        assert_array_almost_equal(h3.data,  [0.5,1,1.5])

        h3 = copy(h1)
        try:
            h3 /= h1
        except OverflowError:
            assert True
        else:
            assert False

        h3 = h1.copy(np.float64)
        h3 /= h2
        assert_array_equal(h1.data, np.array([1,2,3],dtype=np.int64))
        assert_array_almost_equal(h3.data,  [0.5,2.0,np.inf])

    def test_rtruediv(self):
        h1 = Histogram(3,[0,10],data=[1,2,3])
        h = 2 / h1
        assert_array_almost_equal(h.data, [2.0, 1.0, 0.666666666667])

    def test_interpolate_nonfinites_1d(self):
        h = Histogram(5,[0,1],data=[1,2,np.nan,4,5],dtype=np.float64)
        h.interpolate_nonfinites()
        assert_array_almost_equal(h.data, [1,2,3,4,5])

        h[2] = np.inf
        h.interpolate_nonfinites()
        assert_array_almost_equal(h.data, [1,2,3,4,5])

        h.uncert = h.uncert
        h.uncert[2] = np.nan
        h.interpolate_nonfinites()
        assert_array_almost_equal(h.data, [1,2,3,4,5])

        h.uncert[2] = np.inf
        h.interpolate_nonfinites()
        assert_array_almost_equal(h.data, [1,2,3,4,5])

    def test_interpolate_nonfinites_2d(self):
        h = Histogram(3,[0,1],5,[0,1],dtype=np.float)
        data = [[1,2,3,4,5],
                [2,3,4,5,6],
                [3,4,5,6,7]]
        h[:] = data[:]
        h[1,1] = np.nan
        h[1,3] = np.inf
        h.interpolate_nonfinites()
        assert_array_almost_equal(h.data, data)

    def test_smooth_1d(self):
        h = Histogram(5,[0,1],data=[1,2,100,4,5])
        h.smooth()
        self.assertEqual(h.data.dtype, np.dtype('float64'))
        assert_array_almost_equal(h.data,
            [3.800462, 13.767176, 70.848758, 15.704054, 7.436677])

        h[:] = [1,2,100,4,5]
        h.uncert = h.data
        h.smooth(1)
        self.assertEqual(h.data.dtype, np.dtype('float64'))
        assert_array_almost_equal(h.data,
             [  6.600924,  25.534353,  41.697517,  27.408108,   9.873355])
        assert_array_almost_equal(h.uncert,
             [  6.600924,  25.534353,  41.697517,  27.408108,   9.873355])

        h[:] = [1,2,100,4,5]
        h.smooth(0)
        self.assertEqual(h.data.dtype, np.dtype('float64'))
        assert_array_almost_equal(h.data, [1,2,100,4,5])

    def test_smooth_2d(self):
        h = Histogram(3,[0,1],5,[0,1])
        data = [[1,2,3,4,5],
                [2,3,4,5,6],
                [3,4,5,6,7]]
        h[:] = data[:]
        h[1,1] -= 3
        h[1,3] += 3
        h.uncert = h.data
        h.smooth()
        assert_array_almost_equal(
            h.data,
            [[ 1.275218,  2.085901,  3.179543,  4.273184,  5.083867],
             [ 2.039745,  1.325137,  4.      ,  6.674863,  5.960255],
             [ 2.916133,  3.726816,  4.820457,  5.914099,  6.724782]])
        assert_array_almost_equal(
            h.uncert,
            [[ 1.275218,  2.085901,  3.179543,  4.273184,  5.083867],
             [ 2.039745,  1.325137,  4.      ,  6.674863,  5.960255],
             [ 2.916133,  3.726816,  4.820457,  5.914099,  6.724782]])

        data = [[1,2,3,4,5],
                [2,3,4,5,6],
                [3,4,5,6,7]]
        h[:] = data[:]
        h[1,1] -= 3
        h[1,3] += 3
        h.smooth(1)
        assert_array_almost_equal(
            h.data,
            [[ 1.550436,  2.171802,  3.359085,  4.546368,  5.167734],
             [ 2.07949 ,  2.650273,  4.      ,  5.349727,  5.92051 ],
             [ 2.832266,  3.453632,  4.640915,  5.828198,  6.449564]])

    def test_slices_data(self):
        h = Histogram(3,[0,1],5,[0,1])
        h.data = [[1,2,3,4,5],
                  [2,3,4,5,6],
                  [3,4,5,6,7]]
        xslices = list(h.slices_data(0))
        yslices = list(h.slices_data(1))
        assert_array_equal(xslices, h.data)
        assert_array_equal(yslices, h.data.T)

    def test_slices_uncert(self):
        h = Histogram(3,[0,1],5,[0,1])
        h.uncert = [[1,2,3,4,5],
                    [2,3,4,5,6],
                    [3,4,5,6,7]]
        xslices = list(h.slices_uncert(0))
        yslices = list(h.slices_uncert(1))
        assert_array_equal(xslices, h.uncert)
        assert_array_equal(yslices, h.uncert.T)

    def test_slices(self):
        h = Histogram(3,[0,1],5,[0,1],'y','data')
        h.data = [[1,2,3,4,5],
                  [2,3,4,5,6],
                  [3,4,5,6,7]]
        itr = h.slices(0)
        hs = next(itr)
        hexpect = Histogram(5,[0,1],'y','data',data=[1,2,3,4,5])
        self.assertTrue(hs.isidentical(hexpect))

        hs = next(itr)
        hexpect = Histogram(5,[0,1],'y','data',data=[2,3,4,5,6])
        self.assertTrue(hs.isidentical(hexpect))

        itr = h.slices(1)
        hs = next(itr)
        hexpect = Histogram((3,[0,1]),'data',data=[1,2,3])
        self.assertTrue(hs.isidentical(hexpect))

        hs = next(itr)
        hexpect = Histogram((3,[0,1]),'data',data=[2,3,4])
        self.assertTrue(hs.isidentical(hexpect))

        h.uncert = h.uncert
        itr = h.slices(1)
        hs = next(itr)
        hs = next(itr)
        hexpect = Histogram((3,[0,1]),'data',data=[2,3,4],
                            uncert=np.sqrt([2,3,4]))
        self.assertTrue(hs.isidentical(hexpect))

        hslices = list(h.slices(0))
        hexpect = Histogram(5,[0,1],'y','data',data=[1,2,3,4,5])
        self.assertTrue(hslices[0].isidentical(hexpect))
        hexpect.data[:] += 1
        self.assertTrue(hslices[1].isidentical(hexpect))
        hexpect.data[:] += 1
        self.assertTrue(hslices[2].isidentical(hexpect))

    def test_rebin_1d(self):
        h = Histogram(10,[0,10])
        h.set(1)
        hexpect = Histogram(5,[0,10])
        hexpect.set(2)
        self.assertFalse(h.has_uncert)
        hrebin = h.rebin(2)

        self.assertTrue(hrebin.isidentical(hexpect))
        assert_array_almost_equal(hrebin.data, hexpect.data)
        assert_array_almost_equal(hrebin.axes[0].edges, hexpect.axes[0].edges)

        hexpect = Histogram(3,[0,9])
        hexpect.set(3)
        hrebin = h.rebin(3)
        self.assertTrue(hrebin.isidentical(hexpect))

        hexpect = Histogram(3,[1,10])
        hexpect.set(3)
        hrebin = h.rebin(3, snap='high')
        self.assertTrue(hrebin.isidentical(hexpect))

        hexpect = Histogram([0,1,4,7,10])
        hexpect.data = [1,3,3,3]
        hrebin = h.rebin(3, snap='high', clip=False)
        self.assertTrue(hrebin.isidentical(hexpect))

        hexpect = Histogram([0,3,6,9,10])
        hexpect.data = [3,3,3,1]
        hrebin = h.rebin(3, snap='low', clip=False)
        self.assertTrue(hrebin.isidentical(hexpect))

        h.uncert = np.ones(h.shape, dtype=np.float64)
        hexpect = Histogram(3,[0,9])
        hexpect.set(3)
        hexpect.uncert = [np.sqrt(3)]*3
        hrebin = h.rebin(3)
        self.assertTrue(hrebin.isidentical(hexpect))

    def test_rebin_2d(self):
        h = Histogram(3,[0,3],6,[0,6])
        h.set(1)

        hexpect = Histogram(3,[0,3],2,[0,6])
        hexpect.set(3)
        hrebin = h.rebin(3, 1)
        self.assertTrue(hrebin.isidentical(hexpect))

        hexpect = Histogram(1,[0,3],6,[0,6])
        hexpect.set(3)
        hrebin = h.rebin(3)
        self.assertTrue(hrebin.isidentical(hexpect))

        hexpect = Histogram(1,[0,2],6,[0,6])
        hexpect.data = [[2,2,2,2,2,2]]
        hrebin = h.rebin(2)
        self.assertTrue(hrebin.isidentical(hexpect))

        hexpect = Histogram(1,[1,3],6,[0,6])
        hexpect.data = [[2,2,2,2,2,2]]
        hrebin = h.rebin(2, snap='high')
        self.assertTrue(hrebin.isidentical(hexpect))

    def test_rebin_2d_noclip(self):
        h = Histogram(3,[0,3],6,[0,6])
        h.set(1)

        hexpect = Histogram([0,2,3],6,[0,6])
        hexpect.data = [[2,2,2,2,2,2],[1,1,1,1,1,1]]
        hrebin = h.rebin(2, snap='low', clip=False)
        self.assertTrue(hrebin.isidentical(hexpect))

        hexpect = Histogram([0,1,3],6,[0,6])
        hexpect.data = [[1,1,1,1,1,1],[2,2,2,2,2,2]]
        hrebin = h.rebin(2, snap='high', clip=False)
        self.assertTrue(hrebin.isidentical(hexpect))

    def test_cut_1d(self):
        h1 = Histogram(100,[0,10])

        h1a = h1.cut(-3,3)
        h1c = h1.cut(1,3)
        h1b = h1.cut(0,3)
        h1d = h1.cut(3,10)
        h1e = h1.cut(3,20)

        h1 = Histogram(10,[0,10])
        h1.data = np.linspace(0,9,10)

        h1a = h1.cut(-3,3)
        assert_array_almost_equal(h1a.axes[0].edges,[0,1,2,3])
        assert_array_almost_equal(h1a.data,[0,1,2])

        h1c = h1.cut(1,3)
        assert_array_almost_equal(h1c.axes[0].edges,[1,2,3])
        assert_array_almost_equal(h1c.data,[1,2])

        h1b = h1.cut(0,3)
        assert_array_almost_equal(h1b.axes[0].edges,[0,1,2,3])
        assert_array_almost_equal(h1b.data,[0,1,2])

        h1d = h1.cut(3,10)
        assert_array_almost_equal(h1d.axes[0].edges,[3,4,5,6,7,8,9,10])
        assert_array_almost_equal(h1d.data,[3,4,5,6,7,8,9])

        h1e = h1.cut(3,20)
        assert_array_almost_equal(h1e.axes[0].edges,[3,4,5,6,7,8,9,10])
        assert_array_almost_equal(h1e.data,[3,4,5,6,7,8,9])

        h1.uncert = [2]*len(h1.data)
        h1e = h1.cut(3,20)
        assert_array_almost_equal(h1e.axes[0].edges,[3,4,5,6,7,8,9,10])
        assert_array_almost_equal(h1e.data,[3,4,5,6,7,8,9])
        assert_array_almost_equal(h1e.uncert,[2]*7)

    def test_cut_1d_alt(self):
        h = Histogram(10,[0,10],data=np.linspace(0,9,10))

        hcut = h.cut(7)
        assert_array_almost_equal(hcut.axes[0].edges, [7,8,9,10])
        assert_array_almost_equal(hcut.data, [7,8,9])

        hcut = h.cut((7,))
        assert_array_almost_equal(hcut.axes[0].edges, [7,8,9,10])
        assert_array_almost_equal(hcut.data, [7,8,9])

        hcut = h.cut(None,3)
        assert_array_almost_equal(hcut.axes[0].edges, [0,1,2,3])
        assert_array_almost_equal(hcut.data, [0,1,2])

        hcut = h.cut((None,3))
        assert_array_almost_equal(hcut.axes[0].edges, [0,1,2,3])
        assert_array_almost_equal(hcut.data, [0,1,2])

    def test_cut_2d(self):
        h2 = Histogram((20,[0,10]),(12,[-3,3]))
        h2.set(1)
        h2a = h2.cut((-1,1),axis=0)
        hexpect = Histogram(2,[0,1],12,[-3,3])
        hexpect.set(1)
        self.assertTrue(h2a.isidentical(hexpect))

        h2b = h2.cut((-1,1),axis=1)
        hexpect = Histogram(2,[0,1],12,[-3,3])
        hexpect.set(1)
        self.assertTrue(h2a.isidentical(hexpect))

        h3 = Histogram((100,[-30,330]), (100,[-50,50]))
        h3a = h3.cut(-30,30,axis=0)
        h3b = h3.cut(270,330,axis=0)

    def test_occupancy(self):
        h = Histogram(10,[0,10])
        h.fill([1,1,1,2,2,2,3])
        hocc = h.occupancy(4,[-0.5,3.5])
        assert_array_almost_equal(hocc.data, [7,1,0,2])

        hocc = h.occupancy(4)
        assert_array_almost_equal(hocc.data, [7,1,0,2])

    def test_fit_1d(self):
        h = Histogram(10,[0,10])
        popt, pcov, ptest = h.fit(lambda x,*p: np.poly1d(p)(x), [1])
        self.assertEqual(popt.shape, (1,))
        self.assertEqual(pcov.shape, (1,1))
        self.assertEqual(len(ptest), 2)
        self.assertAlmostEqual(popt[0], 0)
        self.assertAlmostEqual(pcov[0,0], 0)
        self.assertAlmostEqual(ptest[0], 0)
        self.assertAlmostEqual(ptest[1], 1)

        h.data[...] = 1
        popt, pcov, ptest = h.fit(lambda x,*p: np.poly1d(p)(x), [1])
        self.assertEqual(popt.shape, (1,))
        self.assertEqual(pcov.shape, (1,1))
        self.assertEqual(len(ptest), 2)
        self.assertAlmostEqual(popt[0], 1)
        self.assertAlmostEqual(pcov[0,0], 0)
        self.assertAlmostEqual(ptest[0], 0)
        self.assertAlmostEqual(ptest[1], 1)

        h.data = np.linspace(3,20,len(h.data))
        popt, pcov, ptest = h.fit(lambda x,*p: np.poly1d(p)(x), [1])
        self.assertAlmostEqual(popt[0], 7.70524386)
        self.assertAlmostEqual(pcov[0,0], 0.77052438)
        self.assertAlmostEqual(ptest[0], 6.1739150908012181)
        self.assertAlmostEqual(ptest[1], 0)

        h = Histogram(2,[0,1])
        h.data[...] = [1,2]
        h.uncert = [0,1]
        popt, pcov, ptest = h.fit(lambda x,*p: np.poly1d(p)(x), [1])
        self.assertAlmostEqual(popt[0], 2)
        self.assertAlmostEqual(pcov[0,0], 1)
        self.assertTrue(np.isnan(ptest[0]))
        self.assertTrue(np.isnan(ptest[1]))

        h = Histogram(2,[0,1])
        h.data[...] = [1,2]
        h.uncert = [10000,1]
        popt, pcov, ptest = h.fit(lambda x,*p: np.poly1d(p)(x), [1])
        self.assertAlmostEqual(popt[0], 2)
        self.assertAlmostEqual(pcov[0,0], 1)
        self.assertAlmostEqual(ptest[0], 0.5)
        self.assertTrue(np.isnan(ptest[1]))

    def test_fit_exceptions(self):
        h = Histogram(2,[0,1],data=[3,5],uncert=[3,10], dtype=float)
        with self.assertRaises(ValueError):
            h.fit(lambda *a: 0, [1], sigma=[1,2])
        with self.assertRaises(ValueError):
            h.fit(lambda *a: 0, [1], absolute_sigma=False)
        h.data = [np.inf, np.nan]
        with self.assertRaises(ValueError):
            h.fit(lambda *a: 0, [1])
        h.data = [0,1]
        with self.assertRaises(RuntimeError):
            h.fit(lambda *a: 0, [1])

    def test_fit_p0_func(self):
        def p0(hist):
            return [hist.mean()[0].nominal_value]

        h = Histogram(2, [0,1], data=[1,2], uncert=[0,1])
        popt, pcov, ptest = h.fit(lambda x,*p: np.poly1d(p)(x), p0)

        self.assertAlmostEqual(popt[0], 2)
        self.assertAlmostEqual(pcov[0,0], 1)
        self.assertTrue(np.isnan(ptest[0]))
        self.assertTrue(np.isnan(ptest[1]))

    def test_fit_ptest(self):
        poly = lambda x,*p: np.poly1d(p)(x)
        h = Histogram(4, [0,1], data=[2,7,4,5], uncert=[0.4,0.2,0.5,0.5])
        popt, pcov, ptest = h.fit(poly, [1,1], test='ktest')
        self.assertAlmostEqual(ptest[0], 0.559980109)
        self.assertAlmostEqual(ptest[1], 0.1057123562)

        popt, pcov, ptest = h.fit(poly, [1,1], test='shapiro')
        self.assertAlmostEqual(ptest[0], 0.9251196384429932,)
        self.assertAlmostEqual(ptest[1], 0.5660210251808167)

        popt, pcov, ptest = h.fit(poly, [1,1], test='chisquare')
        self.assertAlmostEqual(ptest[0], 1.6721828728355184)
        self.assertAlmostEqual(ptest[1], 0.067435462790183101)



if __name__ == '__main__':
    from . import main
    main()
