import unittest

import numpy as np

from histogram import HistogramAxis

class TestHistogramAxis(unittest.TestCase):

    def test___eq__(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.__eq__(that, tol))
        assert True # TODO: implement your test here

    def test___getitem__(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.__getitem__(i))
        assert True # TODO: implement your test here

    def test___init__(self):
        a1 = HistogramAxis(np.linspace(0,10,100))
        a2 = HistogramAxis(100,[0,10])
        a3 = HistogramAxis(100,[0,10],r'label')
        a4 = HistogramAxis(a3)

    def test___str__(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.__str__())
        assert True # TODO: implement your test here

    def test_bin(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.bin(x))
        assert True # TODO: implement your test here

    def test_bincenters(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.bincenters())
        assert True # TODO: implement your test here

    def test_binwidth(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.binwidth(b))
        assert True # TODO: implement your test here

    def test_binwidths(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.binwidths())
        assert True # TODO: implement your test here

    def test_clone(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.clone())
        assert True # TODO: implement your test here

    def test_cut(self):
        a = HistogramAxis(10,[0,10])
        self.assertTrue(np.allclose(a.cut(-1,3)[0].edges, np.linspace(0,3,4))   )
        self.assertTrue(np.allclose(a.cut(7,11)[0].edges, np.linspace(6,10,5))  )
        self.assertTrue(np.allclose(a.cut(-0.1,3)[0].edges, np.linspace(0,3,4)) )
        self.assertTrue(np.allclose(a.cut( 0.0,3)[0].edges, np.linspace(0,3,4)) )
        self.assertTrue(np.allclose(a.cut( 0.1,3)[0].edges, np.linspace(0,3,4)) )
        self.assertTrue(np.allclose(a.cut(2.9,6.9)[0].edges, np.linspace(2,7,6)))
        self.assertTrue(np.allclose(a.cut(3.0,7.0)[0].edges, np.linspace(2,7,6)))
        self.assertTrue(np.allclose(a.cut(3.1,7.1)[0].edges, np.linspace(3,8,6)))

    def test_edge_index(self):
        a = HistogramAxis(10,[0,10])
        self.assertAlmostEqual(a.edge_index(-1),0)
        self.assertAlmostEqual(a.edge_index(-1,'nearest'),0)
        self.assertAlmostEqual(a.edge_index(-1,'high'),0)
        self.assertAlmostEqual(a.edge_index(-1,'low'),0)
        self.assertAlmostEqual(a.edge_index(11),10)
        self.assertAlmostEqual(a.edge_index(11,'nearest'),10)
        self.assertAlmostEqual(a.edge_index(11,'high'),10)
        self.assertAlmostEqual(a.edge_index(11,'low'),10)
        self.assertAlmostEqual(a.edge_index(1.9),2)
        self.assertAlmostEqual(a.edge_index(1.9,'nearest'),2)
        self.assertAlmostEqual(a.edge_index(1.9,'high'),2)
        self.assertAlmostEqual(a.edge_index(1.9,'low'),1)
        self.assertAlmostEqual(a.edge_index(2),2)
        self.assertAlmostEqual(a.edge_index(2,'nearest'),2)
        self.assertAlmostEqual(a.edge_index(2,'high'),2)
        self.assertAlmostEqual(a.edge_index(2,'low'),1)
        self.assertAlmostEqual(a.edge_index(2.1),2)
        self.assertAlmostEqual(a.edge_index(2.1,'nearest'),2)
        self.assertAlmostEqual(a.edge_index(2.1,'high'),3)
        self.assertAlmostEqual(a.edge_index(2.1,'low'),2)

    def test_isuniform(self):
        h1 = HistogramAxis(100,[-5,5])
        h2 = HistogramAxis(np.logspace(0,4,100))
        self.assertTrue(h1.isuniform())
        self.assertFalse(h2.isuniform())

    def test_max(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.max())
        assert True # TODO: implement your test here

    def test_min(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.min())
        assert True # TODO: implement your test here

    def test_nbins(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.nbins())
        assert True # TODO: implement your test here

    def test_overflow(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.overflow())
        assert True # TODO: implement your test here

    def test_range(self):
        # histogram_axis = HistogramAxis(bins, range, label)
        # self.assertEqual(expected, histogram_axis.range())
        assert True # TODO: implement your test here

    def test_mergebins(self):
        a = HistogramAxis(10,[0,10])
        self.assertTrue(np.allclose(a.mergebins(2).edges,np.linspace(0,10,6)))
        self.assertTrue(np.allclose(a.mergebins(3).edges,np.linspace(0,9,4)))

if __name__ == '__main__':
    unittest.main()
