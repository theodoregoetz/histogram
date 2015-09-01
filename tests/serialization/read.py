import platform
from histogram import Histogram
from data import h

if platform.python_version().split('.')[0] == '2':
    h.title = h.title.decode('utf-8')
    h.label = h.label.decode('utf-8')
    for ax in h.axes:
        ax.label = ax.label.decode('utf-8')

hh = Histogram.load('h')

assert h == hh

assert h.label == hh.label
assert h.title == hh.title

for a,aa in zip(h.axes,hh.axes):
    assert a == aa
    assert a.label == aa.label

