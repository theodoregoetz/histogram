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
        edge_grid
        edges
        grid
        has_uncert
        max
        min
        overflow_value
        shape
        size
        uncert_ratio

    **Methods**

    .. autosummary::
        asdict
        asline
        asroot
        binwidth
        copy
        cut
        errorbars
        extent
        fill
        fill_from_sample
        fill_one
        fit
        fromdict
        fromroot
        integral
        interpolate_nonfinites
        isidentical
        isuniform
        load
        load_hdf5
        load_npz
        load_root
        mean
        occupancy
        projection
        projection_data
        rebin
        reset
        save
        save_hdf5
        save_npz
        save_root
        set
        set_infs
        set_nans
        set_nonfinites
        slices
        slices_data
        slices_uncert
        smooth
        std
        sum
        sum_data
        var

    **Operators and Special Methods**

    .. autosummary::
        __add__
        __call__
        __eq__
        __iadd__
        __imul__
        __isub__
        __itruediv__
        __mul__
        __radd__
        __rmul__
        __rsub__
        __rtruediv__
        __str__
        __sub__
        __truediv__

Axes, Shape and Labels
----------------------

Methods specific to the axes of this histogram including size, shape and labels.

.. attribute:: Histogram.axes

    This is a list of the :py:class:`HistogramAxis` objects in the same order
    as the dimensions of the :py:attr:`Histogram.data` attribute. This should
    not typically be changed after constructing the :py:class:`Histogram` in
    the first place.

    Example:

        One can access the underlying axes by the index. Here we create a 2D
        histogram, fill it and change the axes labels afterwards before
        plotting::

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
.. autoattribute:: Histogram.shape
.. autoattribute:: Histogram.size
.. autoattribute:: Histogram.overflow_value

Data, Uncertainty and Statistics
--------------------------------

Access to the filled data, associated uncertainty and various properties of the
histogram.

.. autoattribute:: Histogram.data
.. autoattribute:: Histogram.uncert
.. autoattribute:: Histogram.uncert_ratio
.. automethod:: Histogram.max
.. automethod:: Histogram.min
.. automethod:: Histogram.mean
.. automethod:: Histogram.std
.. automethod:: Histogram.var
.. automethod:: Histogram.sum
.. automethod:: Histogram.sum_data
.. automethod:: Histogram.integral
.. automethod:: Histogram.projection
.. automethod:: Histogram.projection_data
.. automethod:: Histogram.occupancy

Filling Histogram with Data
---------------------------

.. automethod:: Histogram.fill
.. automethod:: Histogram.fill_from_sample
.. automethod:: Histogram.fill_one
.. automethod:: Histogram.set
.. automethod:: Histogram.reset
.. automethod:: Histogram.set_infs
.. automethod:: Histogram.set_nans
.. automethod:: Histogram.set_nonfinites

Transformations
---------------

Methods which generally return a new histogram representing the same data in
some transformed view, sometimes with a loss of information (eg. merged bins).

.. automethod:: Histogram.cut
.. automethod:: Histogram.rebin
.. automethod:: Histogram.smooth

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

Saving, Loading and Type Conversions
------------------------------------

.. automethod:: Histogram.copy
.. automethod:: Histogram.asdict
.. automethod:: Histogram.fromdict
.. automethod:: Histogram.asroot
.. automethod:: Histogram.fromroot
.. automethod:: Histogram.save
.. automethod:: Histogram.load
.. automethod:: Histogram.save_hdf5
.. automethod:: Histogram.load_hdf5
.. automethod:: Histogram.save_npz
.. automethod:: Histogram.load_npz
.. automethod:: Histogram.save_root
.. automethod:: Histogram.load_root

Misc
----

.. autoattribute:: Histogram.label
.. autoattribute:: Histogram.title
.. automethod:: Histogram.interpolate_nonfinites
.. automethod:: Histogram.asline
.. autoattribute:: Histogram.has_uncert
.. automethod:: Histogram.errorbars
.. automethod:: Histogram.isidentical
.. automethod:: Histogram.__eq__
.. automethod:: Histogram.__str__
.. automethod:: Histogram.__call__
