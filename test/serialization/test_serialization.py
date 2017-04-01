# coding: utf-8
from __future__ import unicode_literals

import os
import sys
import unittest
import warnings

from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import patch, Mock

if sys.version_info.major < 3:
    import __builtin__ as builtins
else:
    import builtins

from histogram import rc as histrc
from histogram import Histogram, save_histograms, load_histograms


class TestSerialization(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('always')

    def test_prefix(self):
        histrc.overwrite.overwrite = 'always'
        with TemporaryDirectory() as dtmp:
            h = Histogram(3, [0, 1], 'abcθ', 'αβγ', 'χabc')
            h.save('htmp', dtmp)
            htmp = Histogram.load('htmp', dtmp)
            self.assertTrue(h.isidentical(htmp))

            histrc.histdir = os.path.join(dtmp, 'test')
            ftmp = os.path.join(dtmp, 'h1.hist')
            h.save(ftmp)
            htmp = Histogram.load(os.path.join(dtmp, 'h1.hist'))
            self.assertTrue(h.isidentical(htmp))

            h.save('h2.hist')
            htmp = Histogram.load('h2.hist')
            self.assertTrue(h.isidentical(htmp))
            htmp = Histogram.load(os.path.join(dtmp, 'test', 'h2.hist'))
            self.assertTrue(h.isidentical(htmp))

        histrc.overwrite.overwrite = 'ask'

    def test_no_h5py(self):
        orig_import = __import__

        def import_mock(name, *args, **kwargs):
            if name == 'h5py':
                raise ImportError
            else:
                return orig_import(name, *args, **kwargs)

        del sys.modules['histogram']
        del sys.modules['histogram.serialization']
        del sys.modules['histogram.serialization.serialization']
        del sys.modules['histogram.serialization.histogram_hdf5']

        with patch.object(builtins, '__import__', side_effect=import_mock):
            with warnings.catch_warnings(record=True) as w:
                from histogram.serialization import serialization
                self.assertEqual(len(w), 1)
                self.assertRegex(str(w[-1].message), 'Could not import h5py.')
                self.assertFalse(serialization.have_h5py)

        del sys.modules['histogram']
        del sys.modules['histogram.serialization']
        del sys.modules['histogram.serialization.serialization']
        from histogram.serialization import serialization

    def test_no_root(self):
        orig_import = __import__

        def import_mock(name, *args, **kwargs):
            if name == 'ROOT':
                raise ImportError
            else:
                return orig_import(name, *args, **kwargs)

        del sys.modules['histogram']
        del sys.modules['histogram.serialization']
        del sys.modules['histogram.serialization.serialization']
        del sys.modules['histogram.serialization.histogram_root']

        with patch.object(builtins, '__import__', side_effect=import_mock):
            with warnings.catch_warnings(record=True) as w:
                from histogram.serialization import serialization
                self.assertEqual(len(w), 1)
                self.assertRegex(str(w[-1].message), 'Could not import ROOT.')
                self.assertFalse(serialization.have_pyroot)

        del sys.modules['histogram']
        del sys.modules['histogram.serialization']
        del sys.modules['histogram.serialization.serialization']
        from histogram.serialization import serialization


if __name__ == '__main__':
    from .. import main
    main()
