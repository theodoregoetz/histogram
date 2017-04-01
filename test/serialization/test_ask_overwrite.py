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

from histogram import rc
from histogram.serialization import ask_overwrite


class TestAskOverwrite(unittest.TestCase):
    def test_new_file(self):
        self.assertTrue(ask_overwrite.ask_overwrite('non_existant_file'))

    def test_existing_file(self):
        with NamedTemporaryFile() as ftmp:
            with patch('sys.stdout.write', Mock()):
                with patch.object(ask_overwrite, 'input',
                                  Mock(return_value='n')):
                    self.assertFalse(ask_overwrite.ask_overwrite(ftmp.name))


if __name__ == '__main__':
    from .. import main
    main()
