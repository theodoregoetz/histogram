from histogram import Histogram
from data import h

hh = Histogram.load('h')

assert h == hh

assert h.label == hh.label
assert h.title == hh.title

for a,aa in zip(h.axes,hh.axes):
    assert a == aa
    assert a.label == aa.label

