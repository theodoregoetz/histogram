from __future__ import division

import unittest

import numpy as np

from histogram.detail.run_control import RunControl

class TestHistogram(unittest.TestCase):
    def setUp(self):
        pass

    def test(self):
        rc = RunControl()
        rc.unlock()

        rc.a.b = 1

        print(rc)

        rc.lock()

        #rc.c = 3
        #rc.c.d = 3

        print('\n')
        print(rc)

if __name__ == '__main__':
    test = TestHistogram()
    test.test()
