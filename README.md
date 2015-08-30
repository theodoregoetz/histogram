histogram
=========

A histogram object written in Python for scientific data-reduction and statistical analysis

The primary object of this module is a continuous-domain [histogram](https://en.wikipedia.org/wiki/Histogram). Any number of dimensions is supported though only the lower dimensions can be visualized. For more information, see the following documentation:

* **[wiki pages](https://github.com/theodoregoetz/histogram/wiki)**
* **[API Reference](http://theodoregoetz.github.io/histogram)**

This package is dependent on [NumPy and SciPy](http://www.scipy.org), provides methods for producing and showing graphics through [Matplotlib](http://matplotlib.org) and can be serialized via python's pickling, NumPy's binary format or [HDF5](https://www.hdfgroup.org).

Quick Start
-----------

In this module, the histogram is elevated to its own object, effectively
merging the array and associated edges returned by the
`numpy.histogramdd()` method. Creating, filling and displaying a histogram
should be as simple as this example:

```python
import numpy as np
from matplotlib import pyplot
from histogram import Histogram, plothist

np.random.seed(1)

h = Histogram(100,[0,10])
h.fill(np.random.normal(5,1,10000))
plothist(h)

pyplot.show()
```

![Example Histogram](https://raw.githubusercontent.com/wiki/theodoregoetz/histogram/images/home_ex01.png)

Similar Packages and Software
-----------------------------

While there is no lack of other histogramming libraries out there, most have a narrow focus or seem to be quickly put together, lacking documentation, or are part of a much larger analysis framework. This goal of this project is to be a solution for all python developers who want more functionality than that provided by NumPy's `histogramdd()` method.

This project took many cues from the following python packages, trying to merge the features of each one into a single object:

* [histogram](https://pypi.python.org/pypi/histogram)
* [pyhistogram](https://pypi.python.org/pypi/pyhistogram)
* [histogramy](https://pypi.python.org/pypi/histogramy)
* [pypeaks](https://pypi.python.org/pypi/pypeaks)
* [SimpleHist](https://pypi.python.org/pypi/SimpleHist)
* [hist](https://pypi.python.org/pypi/hist)
* [hierogram](https://pypi.python.org/pypi/hierogram)
* [histo](https://pypi.python.org/pypi/histo)
* [python-metrics](https://pypi.python.org/pypi/python-metrics)
* [statscounter](https://pypi.python.org/pypi/statscounter)
* [multihist](https://pypi.python.org/pypi/multihist)
* [vaex](https://pypi.python.org/pypi/vaex)
* [datagram](https://pypi.python.org/pypi/datagram)
* [hdrhistogram](https://pypi.python.org/pypi/hdrhistogram)

Furthermore, the author was greatly influenced by the histogram classes
found in [CERN's ROOT data analysis framework](https://root.cern.ch).
