import unittest

from histogram.detail import RunControl, skippable


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
        rc.plot.baseline = 'bottom'
        rc.plot.patch.alpha = 0.6

        rc.lock()

        rc_str = '''\
fill_type = 'int'
plot.baseline = 'bottom'
plot.patch.alpha = 0.6'''

        self.assertEqual(rc_str, str(rc))
        rc.clear()
        rc.update(**rc_saved)


class TestIterArgs(unittest.TestCase):
    def test_skippable(self):
        val = 0
        itr = iter(skippable([0,1,2,3]))
        itr(1)
        itr, a = next(itr)
        itr, a = itr.next()
        self.assertEqual(a, 2)

        for skip, arg in skippable([0,1,2,3]):
            self.assertEqual(arg, val)
            skip(1)
            val += 2

if __name__ == '__main__':
    from . import main
    main()
