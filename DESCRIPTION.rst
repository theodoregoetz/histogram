Histogram
=========

A `Histogram` object tightly coupled to the NumPy and SciPy modules.

This project tries to combine most of the best parts of the following modules available through the PyPI repository:

    * histogram
    * pyhistogram
    * histogramy
    * pypeaks
    * SimpleHist
    * hist
    * heriogram
    * histo
    * python-metrics
    * statscounter
    * multihist
    * vaex
    * datagram
    * hdrhistogram

Clearly, there have been many attempts to provide a useable Histogram class/object. Some are better than others but each one has at least one feature which I thought was worthy of a truly general python module. Like `multihist` in particular, this is essentially a wrapper around `numpy`'s `histogramdd()` method. My long-term goal to have this class incorporated into either `SciPy` or `Panda`.
