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
                self.assertFalse(serialization.HAVE_H5PY)

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
                self.assertFalse(serialization.HAVE_PYROOT)

            with self.assertRaises(ImportError):
                Histogram(3,[0,1]).save('tmp.root')
            with patch('os.path.exists', Mock(return_value=True)):
                with self.assertRaises(ImportError):
                    Histogram.load('tmp.root')

        del sys.modules['histogram']
        del sys.modules['histogram.serialization']
        del sys.modules['histogram.serialization.serialization']
        from histogram.serialization import serialization

    def test_no_file(self):
        with patch('os.path.exists', Mock(return_value=False)):
            with self.assertRaises(Exception):
                Histogram.load('tmp.hist')
            with self.assertRaises(Exception):
                load_histograms('tmp.hdf5')


if __name__ == '__main__':
    from .. import main
    main()
