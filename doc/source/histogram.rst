.. py:module:: histogram

Histogram
=========

.. autoclass:: Histogram

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

Axes and Shape
--------------

Methods specific to the axis or axes of this histogram including size and shape of the data.

.. autoattribute:: Histogram.binvolumes
.. automethod:: Histogram.binwidth
.. autoattribute:: Histogram.binwidths
.. autoattribute:: Histogram.dim
.. autoattribute:: Histogram.edge_grid
.. autoattribute:: Histogram.edges
.. autoattribute:: Histogram.grid
.. automethod:: Histogram.isuniform
.. autoattribute:: Histogram.overflow
.. autoattribute:: Histogram.shape
.. autoattribute:: Histogram.size

Modifying Methods
-----------------

Methods which directly modify the data within this histogram.

.. automethod:: Histogram.clear_nans
.. automethod:: Histogram.fill
.. automethod:: Histogram.fill_from_sample
.. automethod:: Histogram.fill_one
.. automethod:: Histogram.set
.. automethod:: Histogram.reset

Data and Statistics
-------------------

Non-modifying methods which give some information about the data or uncertainty contained in this histogram instance.

.. automethod:: Histogram.dtype
.. automethod:: Histogram.errorbars
.. automethod:: Histogram.extent
.. automethod:: Histogram.integral
.. automethod:: Histogram.max
.. automethod:: Histogram.mean
.. automethod:: Histogram.min
.. automethod:: Histogram.std
.. automethod:: Histogram.sum

Transformations
---------------

Methods which generally return a new histogram representing the same data in some transformed view.

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
