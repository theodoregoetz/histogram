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

    def print_help():
        parser._print_help()
        unittest.main()
    parser._print_help = parser.print_help
    parser.print_help = print_help

    args,unknown_args = parser.parse_known_args(sys.argv)

    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    if args.random:
        unittest.defaultTestLoader.sortTestMethodsUsing = \
            lambda *a: random.choice((-1,1))
        def suite_init(self,tests=()):
            self._tests = []
            self._removed_tests = 0
            if isinstance(tests, list):
                random.shuffle(tests)
            self.addTests(tests)
        unittest.defaultTestLoader.suiteClass.__init__ = suite_init

    unittest.main(argv=unknown_args)
