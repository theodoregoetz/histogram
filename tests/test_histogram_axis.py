# -*- coding: utf-8 -*-

import unittest

import numpy as np

from histogram import HistogramAxis

class TestHistogramAxis(unittest.TestCase):

    def isclose(self,a,b):
        assert np.allclose(a,b)

    def test___init__(self):
        a1 = HistogramAxis(np.linspace(0,10,100))
        a2 = HistogramAxis(np.linspace(0,10,100),'label')
        a3 = HistogramAxis(100,[0,10])
        a4 = HistogramAxis(100,[0,10],'label')
        a5 = HistogramAxis(100,[0,10],label='label')
        a6 = HistogramAxis(100,limits=[0,10],label='label')
        a7 = HistogramAxis(100,label='label',limits=[0,10])
        a8 = HistogramAxis(a7)
        a9 = HistogramAxis(a8,label='newlabel')

        self.isclose(a1.edges,np.linspace(0,10,100))
        self.assertEqual(a1.label,None)
        self.isclose(a2.edges,np.linspace(0,10,100))
        self.assertEqual(a2.label,'label')
        self.isclose(a3.edges,np.linspace(0,10,101))
        self.assertEqual(a3.label,None)
        self.isclose(a4.edges,np.linspace(0,10,101))
        self.assertEqual(a4.label,'label')
        self.isclose(a5.edges,np.linspace(0,10,101))
        self.assertEqual(a5.label,'label')
        self.isclose(a6.edges,np.linspace(0,10,101))
        self.assertEqual(a6.label,'label')
        self.isclose(a7.edges,np.linspace(0,10,101))
        self.assertEqual(a7.label,'label')
        self.isclose(a8.edges,np.linspace(0,10,101))
        self.assertEqual(a8.label,'label')
        self.isclose(a9.edges,np.linspace(0,10,101))
        self.assertEqual(a9.label,'newlabel')

    def test___str__(self):
        a3 = HistogramAxis(100,[0,10],'label')
        self.assertEqual(str(a3), str(np.linspace(0,10,101)))

    def test___repr__(self):
        a1 = HistogramAxis(100,[0,10],'label')
        a2 = eval(repr(a1))
        assert a1.isidentical(a2)

    def test___eq__(self):
        a1 = HistogramAxis(np.linspace(0,10,100))
        a2 = HistogramAxis(np.linspace(0,10,100))
        a3 = HistogramAxis(np.logspace(1,4,100))
        assert a1 == a2
        assert a1 != a3

    def test_edges(self):
        a1 = HistogramAxis(100,[0,10],'x')
        a1.edges = np.linspace(0,10,100)
        a1.edges = np.logspace(1,5,100)
        try:
            a1.edges = [2,1]
            assert False
        except ValueError:
            assert True
        try:
            a1.edges = [[1,2],[1,2]]
            raise Exception
        except AssertionError:
            assert True
        except Exception:
            assert False

    def test_label(self):
        a1 = HistogramAxis(100,[0,10],'label ω')
        assert a1.label == 'label ω'
        a1.label = 1
        assert a1.label == '1'

    def test_nbins(self):
        a1 = HistogramAxis(100,[0,10],'label ω')
        assert a1.nbins == 100

    def test_min(self):
        a1 = HistogramAxis(100,[0,10],'x')
        assert a1.min == 0

    def test_max(self):
        a1 = HistogramAxis(100,[0,10],'x')
        assert a1.max == 10

    def test_limits(self):
        a1 = HistogramAxis(100,[0,10],'x')
        assert a1.limits == tuple((0,10))

    def test_binwidths(self):
        a1 = HistogramAxis(100,[0,10],'x')
        a = np.linspace(0,10,101)
        assert np.allclose(a1.binwidths, a[1:] - a[:-1])

        a2 = HistogramAxis(np.logspace(1,5,100),'x')
        a = np.logspace(1,5,100)
        assert np.allclose(a2.binwidths, a[1:] - a[:-1])

    def test_bincenters(self):
        a1 = HistogramAxis(100,[0,10],'x')
        a = np.linspace(0,10,101)
        assert np.allclose(a1.bincenters, 0.5*(a[:-1] + a[1:]))

    def test_overflow(self):
        a1 = HistogramAxis(100,[0,10],'x')
        assert not a1.inaxis(a1.overflow)

    def test_clone(self):
        a1 = HistogramAxis(100,[0,10],'x')
        a2 = a1.clone()
        assert a1 == a2
        assert a1.label == a2.label
        a2.edges = [0,1,2]
        assert a1 != a2
        a2.label = 'y'
        assert a1.label != a2.label

    def test_inaxis(self):
        a1 = HistogramAxis(100,[0,10],'x')
        assert a1.inaxis(0)
        assert not a1.inaxis(10)
        assert a1.inaxis(5)
        assert not a1.inaxis(-1)
        assert not a1.inaxis(11)

    def test_bin(self):
        a1 = HistogramAxis(10,[0,10],'x')
        assert a1.bin(0) == 0
        assert a1.bin(0.5) == 0
        assert a1.bin(1) == 1
        assert a1.bin(9.9999999) == 9

    def test_edge_index(self):
        a = HistogramAxis(10,[0,10])
        self.isclose(a.edge_index(-1),0)
        self.isclose(a.edge_index(-1,'nearest'),0)
        self.isclose(a.edge_index(-1,'high'),0)
        self.isclose(a.edge_index(-1,'low'),0)
        self.isclose(a.edge_index(11),10)
        self.isclose(a.edge_index(11,'nearest'),10)
        self.isclose(a.edge_index(11,'high'),10)
        self.isclose(a.edge_index(11,'low'),10)
        self.isclose(a.edge_index(1.9),2)
        self.isclose(a.edge_index(1.9,'nearest'),2)
        self.isclose(a.edge_index(1.9,'high'),2)
        self.isclose(a.edge_index(1.9,'low'),1)
        self.isclose(a.edge_index(2),2)
        self.isclose(a.edge_index(2,'nearest'),2)
        self.isclose(a.edge_index(2,'high'),3)
        self.isclose(a.edge_index(2,'low'),2)
        self.isclose(a.edge_index(2.1),2)
        self.isclose(a.edge_index(2.1,'nearest'),2)
        self.isclose(a.edge_index(2.1,'high'),3)
        self.isclose(a.edge_index(2.1,'low'),2)

    def test_binwidth(self):
        a = HistogramAxis(10,[0,10])
        assert np.isclose(a.binwidth(), 1)

    def test_cut(self):
        a = HistogramAxis(10,[0,10])
        assert np.allclose(a.cut(-1,3)[0].edges, np.linspace(0,3,4))   
        assert np.allclose(a.cut(7,11)[0].edges, np.linspace(7,10,4))  
        assert np.allclose(a.cut(-0.1,3)[0].edges, np.linspace(0,3,4)) 
        assert np.allclose(a.cut( 0.0,3)[0].edges, np.linspace(0,3,4)) 
        assert np.allclose(a.cut( 0.1,3)[0].edges, np.linspace(0,3,4)) 
        assert np.allclose(a.cut(2.9,6.9)[0].edges, np.linspace(3,7,5))
        assert np.allclose(a.cut(3.0,7.0)[0].edges, np.linspace(3,7,5))
        assert np.allclose(a.cut(3.1,7.1)[0].edges, np.linspace(3,7,5))
        assert np.allclose(a.cut(3.1,7.49999)[0].edges, np.linspace(3,7,5))
        assert np.allclose(a.cut(3.1,7.5)[0].edges, np.linspace(3,8,6))

    def test_isuniform(self):
        h1 = HistogramAxis(100,[-5,5])
        h2 = HistogramAxis(np.logspace(0,4,100))
        assert h1.isuniform()
        assert not h2.isuniform()

    def test_mergebins(self):
        a = HistogramAxis(10,[0,10])
        assert np.allclose(a.mergebins(2).edges,np.linspace(0,10,6))
        assert np.allclose(a.mergebins(3).edges,np.linspace(0,9,4))

if __name__ == '__main__':
    unittest.main()
