# coding: utf-8
from __future__ import unicode_literals

import unittest

from numpy import random as rand

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from histogram import Histogram

from .. import comparator

class TestImage(unittest.TestCase):
    def test_hist_1d(self):

        rand.seed(1)

        h = Histogram(40, [0,1], 'χ (cm)', 'counts', 'Some Data')
        h.fill(rand.normal(0.5, 0.1, 2000))

        fig, ax = plt.subplots()
        pt = ax.plothist(h)

        self.assertTrue(comparator.compare_images(
            fig.savefig, 'hist_1d.png'))


if __name__ == '__main__':
    from .. import main
    main()