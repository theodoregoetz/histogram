import sys

if sys.version_info.major < 3:
    import tempfile
    from backports.tempfile import TemporaryDirectory
    tempfile.TemporaryDirectory = TemporaryDirectory

    import backports.unittest_mock
    backports.unittest_mock.install()

    import unittest

    def assertIsNone(self, obj):
        return self.assertTrue(obj is None)
    unittest.TestCase.assertIsNone = assertIsNone

    if sys.version_info < (3,3):
        def assertRegex(self, text, regexp, msg=None):
            return self.assertRegexpMatches(text, regexp, msg)
        unittest.TestCase.assertRegex = assertRegex

import os
import platform
from subprocess import Popen, PIPE
from .graphics.image_comparison import ImageComparator

def update_baseline_images():
    baseline_directory = 'image_comparisons/baseline'
    repo = 'https://github.com/theodoregoetz/histogram_baseline.git'
    dist = platform.dist()
    ver = sys.version_info
    branch = '{}-{}-py{}'.format(dist[0], dist[1].split('.')[0],
                                 ''.join(str(x) for x in ver[:2]))
    if os.path.exists(os.path.join(baseline_directory, '.git')):
        cmds = ['git fetch --update-head-ok --depth=1 origin {branch}:{branch}',
                'git checkout {branch}']
        for cmd in cmds:
            proc = Popen(cmd.format(branch=branch), shell=True,
                         env=os.environ, cwd=baseline_directory,
                         stdout=PIPE, stderr=PIPE)
            out, err = proc.communicate()
            if proc.returncode != 0:
                return None
    else:
        cmd = 'git clone --depth=1 --branch={branch} {repo} {outdir}'
        cmd = cmd.format(repo=repo, branch=branch, outdir=baseline_directory)
        proc = Popen(cmd, shell=True, env=os.environ, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
        if proc.returncode != 0:
            return None

    return ImageComparator(baseline_directory)

from . import conf

def main():
    import logging
    import os
    import random
    import sys
    import unittest

    from argparse import ArgumentParser, SUPPRESS

    parser = ArgumentParser(usage=SUPPRESS)
    parser.add_argument('-r', '--random',
        action='store_true',
        default=False,
        help='''randomize ordering of test cases and further randomize
                test methods within each test case''')
    parser.add_argument('-d', '--debug',
        action='store_true',
        default=False,
        help='''Set logging output to DEBUG''')
    parser.add_argument('--fast',
        action='store_true',
        default=conf.fast,
        help='''Skip slow unittests (image comparisons).''')

    def print_help():
        parser._print_help()
        unittest.main()

    parser._print_help = parser.print_help
    parser.print_help = print_help

    args, unknown_args = parser.parse_known_args(sys.argv)

    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    conf.fast = args.fast
    if not conf.fast:
        conf.comparator = update_baseline_images()

    if args.random:
        unittest.defaultTestLoader.sortTestMethodsUsing = \
            lambda *a: random.choice((-1,1))
        def suite_init(self, tests=()):
            self._tests = []
            self._removed_tests = 0
            if isinstance(tests, list):
                random.shuffle(tests)
            self.addTests(tests)
        unittest.defaultTestLoader.suiteClass.__init__ = suite_init

    unittest.main(argv=unknown_args)
