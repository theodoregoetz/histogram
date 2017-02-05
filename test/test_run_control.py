import unittest

from histogram.detail import RunControl


class TestRunControl(unittest.TestCase):

    def test_lock(self):
        rc = RunControl()
        rc_saved = dict(rc)
        rc.unlock()
        rc.clear()

        rc.lock()

        with self.assertRaises(KeyError):
            rc.a.b = 1

        rc.unlock()
        rc.a.b = 1
        self.assertEqual(rc.a.b, 1)
        rc.lock()
        self.assertEqual(rc.a.b, 1)

        rc.a.b = 2
        self.assertEqual(rc.a.b, 2)

        rc.clear()
        rc.update(**rc_saved)

    def test_str(self):
        rc = RunControl()
        rc_saved = dict(rc)
        rc.unlock()
        rc.clear()

        rc.fill_type = 'int'
        rc.histdir = None
        rc.overwrite.overwrite = 'ask'
        rc.overwrite.timestamp = None
        rc.overwrite.timeout = 30*60
        rc.plot.baseline = 'bottom'
        rc.plot.patch.alpha = 0.6

        rc.lock()

        rc_str = '''\
fill_type = 'int'
histdir = None
overwrite.overwrite = 'ask'
overwrite.timeout = 1800
overwrite.timestamp = None
plot.baseline = 'bottom'
plot.patch.alpha = 0.6'''

        self.assertEqual(rc_str, str(rc))
        rc.clear()
        rc.update(**rc_saved)


if __name__ == '__main__':
    from . import main
    main()
