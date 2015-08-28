.. py:module:: histogram

Histogram
=========

.. autoclass:: Histogram

    **Internal Data Attributes**

    .. autosummary::
        axes
        data
        label
        title
        uncert

    **Attributes**

    .. autosummary::
        binvolumes
        binwidths
        dim
        dtype
        edge_grid
        edges
        grid
        max
        min
        overflow
        shape
        size

    **Methods**

    .. autosummary::
        __add__
        __call__
        __truediv__
        __iadd__
        __itruediv__
        __imul__
        __isub__
        __mul__
        __radd__
        __rtruediv__
        __rmul__
        __rsub__
        __str__
        __sub__
        added_uncert
        added_uncert_ratio
        aspolygon
        binwidth
        clear_nans
        clone
        cut
        cut_data
        errorbars
        extent
        fill
        fill_from_sample
        fill_one
        fit
        integral
        interpolate_nans
        isuniform
        mean
        occupancy
        projection
        rebin
        reset
        set
        slices
        slices_data
        slices_uncert
        smooth
        std
        sum
        sum_over_axes

Axes, Shape and Labels
----------------------

Methods specific to the axes of this histogram including size, shape and labels.

.. attribute:: Histogram.axes

    This is a list of the :py:class:`HistogramAxis` objects in the same order as the dimensions of the :py:attr:`Histogram.data` attribute. This should not typically be changed after constructing the :py:class:`Histogram` in the first place.

    Example:

        One can access the underlying axes by the index. Here we create a 2D histogram, fill it and change the axes labels afterwards before plotting::

            import numpy as np
            from matplotlib import pyplot
            from histogram import Histogram

            datax = np.random.normal(5,1,10000)
            datay = np.random.normal(0,2,10000)

            h = Histogram((100,[0,10]), (40, [-5,5]))
            h.fill(datax,datay)

            h.axes[0].label = 'x'
            h.axes[1].label = 'y'

            fig,ax = pyplot.subplots(figsize=(4,2.5))
            fig.subplots_adjust(left=.15,bottom=.2,right=.9)
            pt,cb = ax.plothist(h)
            pyplot.show()

        .. image:: images/histogram_2dnorm.png

.. autoattribute:: Histogram.binvolumes
.. automethod:: Histogram.binwidth
.. autoattribute:: Histogram.binwidths
.. autoattribute:: Histogram.dim
.. autoattribute:: Histogram.extent
.. autoattribute:: Histogram.edge_grid
.. autoattribute:: Histogram.edges
.. autoattribute:: Histogram.grid
.. automethod:: Histogram.isuniform
.. autoattribute:: Histogram.overflow
.. autoattribute:: Histogram.shape
.. autoattribute:: Histogram.size

Data, Uncertainty and Statistics
--------------------------------

Access to the filled data, associated uncertainty and various properties of the histogram.

.. autoattribute:: Histogram.data
.. autoattribute:: Histogram.uncert
.. automethod:: Histogram.dtype
.. automethod:: Histogram.errorbars
.. automethod:: Histogram.integral
.. automethod:: Histogram.max
.. automethod:: Histogram.mean
.. automethod:: Histogram.min
.. automethod:: Histogram.std
.. automethod:: Histogram.sum

Filling Histogram with Data
---------------------------

.. automethod:: Histogram.fill
.. automethod:: Histogram.fill_from_sample
.. automethod:: Histogram.fill_one
.. automethod:: Histogram.set
.. automethod:: Histogram.reset
.. automethod:: Histogram.clear_nans

Transformations
---------------

Methods which generally return a new histogram representing the same data in some transformed view, sometimes with a loss of information (eg. merged bins).

.. automethod:: Histogram.cut
.. automethod:: Histogram.cut_data
.. automethod:: Histogram.interpolate_nans
.. automethod:: Histogram.occupancy
.. automethod:: Histogram.projection
.. automethod:: Histogram.rebin
.. automethod:: Histogram.smooth
.. automethod:: Histogram.sum_over_axes

Iterating
---------

.. automethod:: Histogram.slices
.. automethod:: Histogram.slices_data
.. automethod:: Histogram.slices_uncert

Fitting
-------

.. automethod:: Histogram.fit

Arithmetic
----------

.. automethod:: Histogram.__iadd__
.. automethod:: Histogram.__radd__
.. automethod:: Histogram.__add__
.. automethod:: Histogram.__isub__
.. automethod:: Histogram.__rsub__
.. automethod:: Histogram.__sub__
.. automethod:: Histogram.__imul__
.. automethod:: Histogram.__rmul__
.. automethod:: Histogram.__mul__
.. automethod:: Histogram.__itruediv__
.. automethod:: Histogram.__rtruediv__
.. automethod:: Histogram.__truediv__

Misc
----

.. automethod:: Histogram.added_uncert
.. automethod:: Histogram.added_uncert_ratio
.. automethod:: Histogram.aspolygon
.. automethod:: Histogram.clone
.. automethod:: Histogram.__str__
.. automethod:: Histogram.__call__
