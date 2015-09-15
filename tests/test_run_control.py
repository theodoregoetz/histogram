import unittest

from histogram.detail import RunControl

class TestRunControl(unittest.TestCase):

    def test_lock(self):
        rc = RunControl()

        try:
            rc.a.b = 1
            assert False, 'run control not locked!'
        except KeyError:
            assert True

        rc.unlock()
        rc.a.b = 1
        assert rc.a.b == 1
        rc.lock()
        assert rc.a.b == 1

        rc.a.b = 2
        assert rc.a.b == 2

    def test_str(self):
        rc = RunControl()

        rc.unlock()
        rc.clear()

        rc.histdir = None
        rc.overwrite.overwrite = 'ask'
        rc.overwrite.timestamp = None
        rc.overwrite.timeout = 30*60
        rc.plot.baseline = 'bottom'
        rc.plot.patch.alpha = 0.6

        rc.lock()

        rc_str = '''\
histdir = None
overwrite.overwrite = 'ask'
overwrite.timeout = 1800
overwrite.timestamp = None
plot.baseline = 'bottom'
plot.patch.alpha = 0.6'''

        assert rc_str == str(rc)


if __name__ == '__main__':
    test = TestRunControl()
    test.test()
