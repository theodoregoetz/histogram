# coding: utf-8
from __future__ import unicode_literals

import unittest
from warnings import simplefilter

from numpy import random as rand

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt


from histogram import Histogram

from .. import conf

class TestHistogramsMpl(unittest.TestCase):
    def setUp(self):
        if conf.fast or conf.comparator is None:
            raise unittest.SkipTest('slow tests')
        simplefilter('ignore')

    def test_hist_1d(self):
        rand.seed(1)
        h = Histogram(40, [0,1], 'χ (cm)', 'counts', 'Some Data')
        h.fill(rand.normal(0.5, 0.1, 2000))
        fig, ax = plt.subplots()
        pt = ax.plothist(h)
        self.assertTrue(conf.comparator.compare_images(
            fig.savefig, 'hist_1d.png'))

    def test_hist_1d_errorbar(self):
        rand.seed(1)
        h = Histogram(40, [0,1], 'χ (cm)', 'counts', 'Some Data')
        h.fill(rand.normal(0.5, 0.1, 300))
        fig, ax = plt.subplots()
        pt = ax.plothist(h, style='errorbar')
        self.assertTrue(conf.comparator.compare_images(
            fig.savefig, 'hist_1d_errorbar.png'))

    def test_hist_1d_line(self):
        rand.seed(1)
        h = Histogram(40, [0,1], 'χ (cm)', 'counts', 'Some Data')
        h.fill(rand.normal(0.5, 0.1, 300))
        fig, ax = plt.subplots()
        pt = ax.plothist(h, style='line')
        self.assertTrue(conf.comparator.compare_images(
            fig.savefig, 'hist_1d_line.png'))

    def test_hist_2d(self):
        rand.seed(1)
        h = Histogram(40, [0,1], 'χ (cm)', 30, [-5,5], 'ψ', 'counts', 'Some Data')
        h.fill(rand.normal(0.5, 0.1, 1000), rand.normal(0,1,1000))
        fig, ax = plt.subplots()
        pt = ax.plothist(h)
        self.assertTrue(conf.comparator.compare_images(
            fig.savefig, 'hist_2d.png'))

    def test_hist_1d_nonuniform(self):
        h = Histogram([0,1,3], data=[1,2])
        fig, ax = plt.subplots(2)
        pt = ax[0].plothist(h, style='polygon')
        pt = ax[0].plothist(h, style='line', color='black', lw=2)
        pt = ax[1].plothist(h, style='errorbar')
        self.assertTrue(conf.comparator.compare_images(
            fig.savefig, 'hist_1d_nonuniform.png'))

    def test_hist_1d_negative(self):
        h = Histogram([0,1,3], data=[-1,2])
        fig, ax = plt.subplots(2)
        pt = ax[0].plothist(h, style='polygon')
        pt = ax[0].plothist(h, style='line', color='black', lw=2)
        pt = ax[1].plothist(h, style='errorbar')
        self.assertTrue(conf.comparator.compare_images(
            fig.savefig, 'hist_1d_negative.png'))




if __name__ == '__main__':
    from .. import main
    main()
