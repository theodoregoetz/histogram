import unittest

from histogram.detail import RunControl

class TestRunControl(unittest.TestCase):

    def test_lock(self):
        rc = RunControl()

        try:
            rc.a.b = 1
            assert False
        except KeyError:
            assert True

        rc.unlock()
        rc.a.b = 1
        assert rc.a.b == 1
        rc.lock()
        assert rc.a.b == 1

        rc.a.b = 2
        assert rc.a.b == 2

if __name__ == '__main__':
    test = TestRunControl()
    test.test()
