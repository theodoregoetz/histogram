# -*- coding: utf-8 -*-

import numpy as np
import unittest

from numpy.testing import assert_array_almost_equal, assert_array_equal

from histogram import HistogramAxis


class TestHistogramAxis(unittest.TestCase):

    def test___init__(self):
        a1 = HistogramAxis(np.linspace(0, 10, 100))
        a2 = HistogramAxis(np.linspace(0, 10, 100), 'label')
        a3 = HistogramAxis(100, [0, 10])
        a4 = HistogramAxis(100, [0, 10], 'label')
        a5 = HistogramAxis(100, [0, 10], label='label')
        a6 = HistogramAxis(100, limits=[0, 10], label='label')
        a7 = HistogramAxis(100, label='label', limits=[0, 10])

        assert_array_almost_equal(a1.edges, np.linspace(0, 10, 100))
        self.assertIsNone(a1.label)
        assert_array_almost_equal(a2.edges, np.linspace(0, 10, 100))
        self.assertEqual(a2.label, 'label')
        assert_array_almost_equal(a3.edges, np.linspace(0, 10, 101))
        self.assertIsNone(a3.label)
        assert_array_almost_equal(a4.edges, np.linspace(0, 10, 101))
        self.assertEqual(a4.label, 'label')
        assert_array_almost_equal(a5.edges, np.linspace(0, 10, 101))
        self.assertEqual(a5.label, 'label')
        assert_array_almost_equal(a6.edges, np.linspace(0, 10, 101))
        self.assertEqual(a6.label, 'label')
        assert_array_almost_equal(a7.edges, np.linspace(0, 10, 101))
        self.assertEqual(a7.label, 'label')

        self.assertRaises(TypeError, HistogramAxis)
        self.assertRaises(TypeError, HistogramAxis, 10, 11)
        self.assertRaises(TypeError, HistogramAxis, 10, 11, 12)
        self.assertRaises((TypeError, ValueError), HistogramAxis, None,
                          [0, 1, 2])
        self.assertRaises((TypeError, ValueError), HistogramAxis, None,
                          [0, 1, 2], 'x')
        self.assertRaises((TypeError, ValueError), HistogramAxis, 'a',
                          [0, 1, 2])
        self.assertRaises((TypeError, ValueError), HistogramAxis, 'a',
                          [0, 1, 2], 'x')
        self.assertRaises((IndexError, ValueError), HistogramAxis, 'a', 'b',
                          'c')

        if __debug__:
            self.assertRaises(ValueError, HistogramAxis, None)
            self.assertRaises(ValueError, HistogramAxis, 0, [0, 10])
            self.assertRaises(ValueError, HistogramAxis, 1, [0, 1, 2])
            self.assertRaises((TypeError, ValueError), HistogramAxis, 'a')
            self.assertRaises((TypeError, ValueError), HistogramAxis, 'a', 'b')
            self.assertRaises(ValueError, HistogramAxis, 1, [0, 1], 3.14)

    def test___str__(self):
        a3 = HistogramAxis(100, [0, 10], 'label')
        self.assertEqual(str(a3), str(np.linspace(0, 10, 101)))

    def test___repr__(self):
        a1 = HistogramAxis(100, [0, 10], 'label')
        a2 = eval(repr(a1))
        self.assertTrue(a1.isidentical(a2))

        a1 = HistogramAxis(100, [0, 10])
        a2 = eval(repr(a1))
        self.assertTrue(a1.isidentical(a2))

    def test___eq__(self):
        a1 = HistogramAxis(np.linspace(0, 10, 100))
        a2 = HistogramAxis(np.linspace(0, 10, 100))
        a3 = HistogramAxis(np.logspace(1, 4, 100))
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, a3)

    def test_isdentical(self):
        a1 = HistogramAxis(np.linspace(0, 10, 100), 'aa')
        a2 = HistogramAxis(np.linspace(0, 10, 100), 'aa')
        a3 = HistogramAxis(np.linspace(1, 10, 100), 'aa')
        a4 = HistogramAxis(np.linspace(0, 11, 100), 'aa')
        a5 = HistogramAxis(np.linspace(0, 10, 101), 'aa')
        a6 = HistogramAxis(np.linspace(0, 10, 100), 'ab')
        self.assertTrue(a1.isidentical(a2))
        self.assertFalse(a1.isidentical(a3))
        self.assertFalse(a1.isidentical(a4))
        self.assertFalse(a1.isidentical(a5))
        self.assertFalse(a1.isidentical(a6))

    def test_edges(self):
        a1 = HistogramAxis(100, [0, 10], 'x')
        a1.edges = np.linspace(0, 10, 100)
        a1.edges = np.logspace(1, 5, 100)
        if __debug__:
            self.assertRaises(ValueError, setattr, a1, 'edges', [0])
            self.assertRaises(ValueError, setattr, a1, 'edges', [2, 1])
            self.assertRaises(ValueError, setattr, a1, 'edges',
                              [[1, 2], [1, 2]])

    def test_label(self):
        a1 = HistogramAxis(100, [0, 10], 'label ω')
        self.assertEqual(a1.label, 'label ω')
        a1.label = 1
        self.assertEqual(a1.label, '1')
        self.assertTrue(hasattr(a1, '_label'))
        del a1.label
        self.assertEqual(a1.label, None)
        self.assertFalse(hasattr(a1, '_label'))

    def test_nbins(self):
        a1 = HistogramAxis(100, [0, 10], 'label ω')
        self.assertEqual(a1.nbins, 100)

    def test_min(self):
        a1 = HistogramAxis(100, [0, 10], 'x')
        self.assertEqual(a1.min, 0)

    def test_mid(self):
        a1 = HistogramAxis(100, [0, 10], 'x')
        self.assertAlmostEqual(a1.mid, 5.0)

    def test_max(self):
        a1 = HistogramAxis(100, [0, 10], 'x')
        self.assertEqual(a1.max, 10)

    def test_limits(self):
        a1 = HistogramAxis(100, [0, 10], 'x')
        self.assertEqual(a1.limits, tuple((0, 10)))

    def test_binwidths(self):
        a1 = HistogramAxis(100, [0, 10], 'x')
        a = np.linspace(0, 10, 101)
        assert_array_almost_equal(a1.binwidths(), a[1:] - a[:-1])

        a2 = HistogramAxis(np.logspace(1, 5, 100), 'x')
        a = np.logspace(1, 5, 100)
        assert_array_almost_equal(a2.binwidths(), a[1:] - a[:-1])

    def test_bincenters(self):
        a1 = HistogramAxis(100, [0, 10], 'x')
        a = np.linspace(0, 10, 101)
        assert_array_almost_equal(a1.bincenters(), 0.5*(a[:-1] + a[1:]))

    def test_overflow_value(self):
        a1 = HistogramAxis(100, [0, 10], 'x')
        self.assertFalse(a1.inaxis(a1.overflow_value))

    def test_clone(self):
        a1 = HistogramAxis(100, [0, 10], 'x')
        a2 = a1.clone()
        self.assertEqual(a1, a2)
        self.assertEqual(a1.label, a2.label)
        a2.edges = [0, 1, 2]
        self.assertNotEqual(a1, a2)
        a2.label = 'y'
        self.assertNotEqual(a1.label, a2.label)

    def test_inaxis(self):
        a1 = HistogramAxis(100, [0, 10], 'x')
        self.assertTrue(a1.inaxis(0))
        self.assertFalse(a1.inaxis(10))
        self.assertTrue(a1.inaxis(5))
        self.assertFalse(a1.inaxis(-1))
        self.assertFalse(a1.inaxis(11))

    def test_bin(self):
        a1 = HistogramAxis(10, [0, 10], 'x')
        self.assertEqual(a1.bin(0), 0)
        self.assertEqual(a1.bin(0.5), 0)
        self.assertEqual(a1.bin(1), 1)
        self.assertEqual(a1.bin(9.9999999), 9)
        assert_array_equal(a1.bin(np.linspace(0.5, 9.5, 10)), range(10))

    def test_edge_index(self):
        a = HistogramAxis(10, [0, 10])
        assert_array_almost_equal(a.edge_index(-1), 0)
        assert_array_almost_equal(a.edge_index(-1, 'nearest'), 0)
        assert_array_almost_equal(a.edge_index(-1, 'high'), 0)
        assert_array_almost_equal(a.edge_index(-1, 'low'), 0)
        assert_array_almost_equal(a.edge_index(-1, 'both'), (0,0))
        assert_array_almost_equal(a.edge_index(11), 10)
        assert_array_almost_equal(a.edge_index(11, 'nearest'), 10)
        assert_array_almost_equal(a.edge_index(11, 'high'), 10)
        assert_array_almost_equal(a.edge_index(11, 'low'), 10)
        assert_array_almost_equal(a.edge_index(11, 'both'), (10,10))
        assert_array_almost_equal(a.edge_index(1.9), 2)
        assert_array_almost_equal(a.edge_index(1.9, 'nearest'), 2)
        assert_array_almost_equal(a.edge_index(1.9, 'high'), 2)
        assert_array_almost_equal(a.edge_index(1.9, 'low'), 1)
        assert_array_almost_equal(a.edge_index(1.9, 'both'), (1,2))
        assert_array_almost_equal(a.edge_index(2), 2)
        assert_array_almost_equal(a.edge_index(2, 'nearest'), 2)
        assert_array_almost_equal(a.edge_index(2, 'high'), 3)
        assert_array_almost_equal(a.edge_index(2, 'low'), 2)
        assert_array_almost_equal(a.edge_index(2, 'both'), (2,3))
        assert_array_almost_equal(a.edge_index(2.0), 2)
        assert_array_almost_equal(a.edge_index(2.0, 'nearest'), 2)
        assert_array_almost_equal(a.edge_index(2.0, 'high'), 3)
        assert_array_almost_equal(a.edge_index(2.0, 'low'), 2)
        assert_array_almost_equal(a.edge_index(2.0, 'both'), (2,3))
        assert_array_almost_equal(a.edge_index(2.1), 2)
        assert_array_almost_equal(a.edge_index(2.1, 'nearest'), 2)
        assert_array_almost_equal(a.edge_index(2.1, 'high'), 3)
        assert_array_almost_equal(a.edge_index(2.1, 'low'), 2)
        assert_array_almost_equal(a.edge_index(2.1, 'both'), (2,3))

        if __debug__:
            with self.assertRaises(ValueError):
                a.edge_index(2.1, 'x')

    def test_binwidth(self):
        a = HistogramAxis(10, [0, 10])
        assert_array_almost_equal(a.binwidth(), 1)

    def test_cut_none(self):
        a = HistogramAxis(10, [0, 10])

        assertaaeq = assert_array_almost_equal

        assertaaeq(a.cut(None, 3)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(7, None)[0].edges, np.linspace(7, 10, 4))

        with self.assertRaises(TypeError):
            a.cut(0,3,0.)

    def test_cut_nearest(self):
        a = HistogramAxis(10, [0, 10])

        assertaaeq = assert_array_almost_equal

        assertaaeq(a.cut(-1, 3)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(7, 11)[0].edges, np.linspace(7, 10, 4))
        assertaaeq(a.cut(-0.1, 3)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(0.0, 3)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(0.1, 3)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(2.9, 6.9)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.0, 7.0)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.1, 7.1)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.1, 7.49999)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.1, 7.5)[0].edges, np.linspace(3, 8, 6))

        if __debug__:
            with self.assertRaises(ValueError):
                a.cut(0,1,'x')

        # 'nearest','expand','low','high','clip'

        snap = 'nearest'
        assertaaeq(a.cut(-1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(7, 11, snap)[0].edges, np.linspace(7, 10, 4))
        assertaaeq(a.cut(-0.1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(0.0, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(0.1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(2.9, 6.9, snap)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.0, 7.0, snap)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.1, 7.1, snap)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.1, 7.49999, snap)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.1, 7.5, snap)[0].edges, np.linspace(3, 8, 6))

    def test_cut_expand(self):
        a = HistogramAxis(10, [0, 10])

        assertaaeq = assert_array_almost_equal

        snap = 'expand'
        assertaaeq(a.cut(-1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(7, 11, snap)[0].edges, np.linspace(7, 10, 4))
        assertaaeq(a.cut(-0.1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(0.0, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(0.1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(2.9, 6.9, snap)[0].edges, np.linspace(2, 7, 6))
        assertaaeq(a.cut(3.0, 7.0, snap)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.1, 7.1, snap)[0].edges, np.linspace(3, 8, 6))
        assertaaeq(a.cut(3.1, 7.49999, snap)[0].edges, np.linspace(3, 8, 6))
        assertaaeq(a.cut(3.1, 7.5, snap)[0].edges, np.linspace(3, 8, 6))

    def test_cut_low(self):
        a = HistogramAxis(10, [0, 10])

        assertaaeq = assert_array_almost_equal

        snap = 'low'
        assertaaeq(a.cut(-1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(7, 11, snap)[0].edges, np.linspace(7, 10, 4))
        assertaaeq(a.cut(-0.1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(0.0, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(0.1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(2.9, 6.9, snap)[0].edges, np.linspace(2, 6, 5))
        assertaaeq(a.cut(3.0, 7.0, snap)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.1, 7.1, snap)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.1, 7.49999, snap)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.1, 7.5, snap)[0].edges, np.linspace(3, 7, 5))

    def test_cut_high(self):
        a = HistogramAxis(10, [0, 10])

        assertaaeq = assert_array_almost_equal

        snap = 'high'
        assertaaeq(a.cut(-1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(7, 11, snap)[0].edges, np.linspace(7, 10, 4))
        assertaaeq(a.cut(-0.1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(0.0, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(0.1, 3, snap)[0].edges, np.linspace(1, 3, 3))
        assertaaeq(a.cut(2.9, 6.9, snap)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.0, 7.0, snap)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.1, 7.1, snap)[0].edges, np.linspace(4, 8, 5))
        assertaaeq(a.cut(3.1, 7.49999, snap)[0].edges, np.linspace(4, 8, 5))
        assertaaeq(a.cut(3.1, 7.5, snap)[0].edges, np.linspace(4, 8, 5))

    def test_cut_clip(self):
        a = HistogramAxis(10, [0, 10])

        assertaaeq = assert_array_almost_equal

        snap = 'clip'
        assertaaeq(a.cut(-1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(7, 11, snap)[0].edges, np.linspace(7, 10, 4))
        assertaaeq(a.cut(-0.1, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(0.0, 3, snap)[0].edges, np.linspace(0, 3, 4))
        assertaaeq(a.cut(0.1, 3, snap)[0].edges, [0.1, 1, 2, 3])
        assertaaeq(a.cut(2.9, 6.9, snap)[0].edges, [2.9, 3, 4, 5, 6, 6.9])
        assertaaeq(a.cut(3.0, 7.0, snap)[0].edges, np.linspace(3, 7, 5))
        assertaaeq(a.cut(3.1, 7.1, snap)[0].edges, [3.1, 4, 5, 6, 7, 7.1])
        assertaaeq(a.cut(3.1, 5.49999, snap)[0].edges, [3.1, 4, 5, 5.49999])
        assertaaeq(a.cut(3.1, 5.5, snap)[0].edges, [3.1, 4, 5, 5.5])

    def test_isuniform(self):
        h1 = HistogramAxis(100, [-5, 5])
        h2 = HistogramAxis(np.logspace(0, 4, 100))
        self.assertTrue(h1.isuniform())
        self.assertFalse(h2.isuniform())

    def test_mergebins(self):
        a = HistogramAxis(10, [0, 10])
        assert_array_almost_equal(a.mergebins().edges, np.linspace(0, 10, 6))
        assert_array_almost_equal(a.mergebins(2).edges, np.linspace(0, 10, 6))
        assert_array_almost_equal(a.mergebins(3).edges, np.linspace(0, 9, 4))

        assertaaeq = assert_array_almost_equal

        a = HistogramAxis([0, 1, 2, 3, 4, 5])
        assertaaeq(a.mergebins(2,'low',True).edges, [0, 2, 4])
        assertaaeq(a.mergebins(2,'low',False).edges, [0, 2, 4, 5])
        assertaaeq(a.mergebins(2,'high',True).edges, [1, 3, 5])
        assertaaeq(a.mergebins(2,'high',False).edges, [0, 1, 3, 5])

        a = HistogramAxis([0, 1, 2, 3, 4])
        assertaaeq(a.mergebins(2,'low',True).edges, [0, 2, 4])
        assertaaeq(a.mergebins(2,'low',False).edges, [0, 2, 4])
        assertaaeq(a.mergebins(2,'high',True).edges, [0, 2, 4])
        assertaaeq(a.mergebins(2,'high',False).edges, [0, 2, 4])

        if __debug__:
            a = HistogramAxis([0, 1])
            with self.assertRaises(ValueError):
                a.mergebins()

            with self.assertRaises(ValueError):
                a.mergebins(2,'x')

    def test_asdict(self):
        a = HistogramAxis(3, [0, 1])
        d = a.asdict()
        b = HistogramAxis.fromdict(d)
        self.assertEqual(a, b)

        a.label = 'test'
        b = HistogramAxis.fromdict(a.asdict())
        self.assertEqual(a, b)


if __name__ == '__main__':
    from . import main
    main()
