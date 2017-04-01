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

            save_histograms({'h': h}, 'hlist1.hdf5', prefix=dtmp)
            hlist = load_histograms('hlist1', prefix=dtmp)
            self.assertTrue(h.isidentical(hlist['h']))

            save_histograms({'h': h}, 'hlist2')
            hlist = load_histograms('hlist2.hdf5')
            self.assertTrue(h.isidentical(hlist['h']))

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
                from histogram.serialization import (
                    serialization, load_histograms, save_histograms)
                self.assertEqual(len(w), 1)
                self.assertRegex(str(w[-1].message), 'Could not import h5py.')
                self.assertFalse(serialization.have_h5py)

            with self.assertRaises(ImportError):
                Histogram(3,[0,1]).save('tmp.h5')
            with self.assertRaises(ImportError):
                Histogram(3,[0,1]).save('tmp.hdf5')
            with patch('os.path.exists', Mock(return_value=True)):
                with self.assertRaises(ImportError):
                    Histogram.load('tmp.h5')
                with self.assertRaises(ImportError):
                    Histogram.load('tmp.hdf5')
                with self.assertRaises(ImportError):
                    load_histograms('tmp.hdf5')
            with self.assertRaises(ImportError):
                save_histograms({'h':Histogram(3,[0,1])}, 'tmp.hdf5')

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

            with self.assertRaises(ImportError):
                Histogram(3,[0,1]).save('tmp.root')
            with patch('os.path.exists', Mock(return_value=True)):
                with self.assertRaises(ImportError):
                    Histogram.load('tmp.root')

        del sys.modules['histogram']
        del sys.modules['histogram.serialization']
        del sys.modules['histogram.serialization.serialization']
        from histogram.serialization import serialization

    def test_no_overwrite(self):
        from histogram.serialization import serialization, save_histograms
        with patch.object(serialization, 'ask_overwrite',
                          Mock(return_value=False)):
            with patch.object(serialization.log, 'warning', Mock()) as log:
                Histogram(3,[0,1]).save('tmp.hist')
                self.assertEqual(log.call_count, 1)
                save_histograms({'h': Histogram(3,[0,1])}, 'h.hdf5')
                self.assertEqual(log.call_count, 2)

    def test_no_file(self):
        with patch('os.path.exists', Mock(return_value=False)):
            with self.assertRaises(Exception):
                Histogram.load('tmp.hist')
            with self.assertRaises(Exception):
                load_histograms('tmp.hdf5')


if __name__ == '__main__':
    from .. import main
    main()
