histogram
=========

A histogram object written in Python for scientific data-reduction and
statistical analysis

The primary object of this module is a continuous-domain
`histogram <https://en.wikipedia.org/wiki/Histogram>`__. Any number of
dimensions is supported though only the lower dimensions can be
visualized. For more information, see the following documentation:

-  **`wiki pages <https://github.com/theodoregoetz/histogram/wiki>`__**
-  **`API Reference <http://theodoregoetz.github.io/histogram>`__**

This package is dependent on `NumPy and SciPy <http://www.scipy.org>`__,
provides methods for producing and showing graphics through
`Matplotlib <http://matplotlib.org>`__ and can be serialized via
python's pickling, NumPy's binary format or
`HDF5 <https://www.hdfgroup.org>`__.

Quick Start
-----------

In this module, the histogram is elevated to its own object, effectively
merging the array and associated edges returned by the
``numpy.histogramdd()`` method. Creating, filling and displaying a
histogram should be as simple as this example:

.. code:: python

    import numpy as np
    from matplotlib import pyplot
    from histogram import Histogram, plothist

    np.random.seed(1)

    h = Histogram(100,[0,10])
    h.fill(np.random.normal(5,1,10000))
    plothist(h)

    pyplot.show()

.. figure:: https://raw.githubusercontent.com/wiki/theodoregoetz/histogram/images/home_ex01.png
   :alt: Example Histogram

   Example Histogram
Similar Packages and Software
-----------------------------

While there is no lack of other histogramming libraries out there, most
have a narrow focus or seem to be quickly put together, lacking
documentation, or are part of a much larger analysis framework. This
goal of this project is to be a solution for all python developers who
want more functionality than that provided by NumPy's ``histogramdd()``
method.

This project took many cues from the following python packages, trying
to merge the features of each one into a single object:

-  `histogram <https://pypi.python.org/pypi/histogram>`__
-  `pyhistogram <https://pypi.python.org/pypi/pyhistogram>`__
-  `histogramy <https://pypi.python.org/pypi/histogramy>`__
-  `pypeaks <https://pypi.python.org/pypi/pypeaks>`__
-  `SimpleHist <https://pypi.python.org/pypi/SimpleHist>`__
-  `hist <https://pypi.python.org/pypi/hist>`__
-  `hierogram <https://pypi.python.org/pypi/hierogram>`__
-  `histo <https://pypi.python.org/pypi/histo>`__
-  `python-metrics <https://pypi.python.org/pypi/python-metrics>`__
-  `statscounter <https://pypi.python.org/pypi/statscounter>`__
-  `multihist <https://pypi.python.org/pypi/multihist>`__
-  `vaex <https://pypi.python.org/pypi/vaex>`__
-  `datagram <https://pypi.python.org/pypi/datagram>`__
-  `hdrhistogram <https://pypi.python.org/pypi/hdrhistogram>`__
-  `dashi <http://www.ifh.de/~middell/dashi/index.html>`__

Furthermore, the author was greatly influenced by the histogram classes
found in `CERN's ROOT data analysis framework <https://root.cern.ch>`__.
