.. py:module:: histogram

HistogramAxis
=============

.. autoclass:: HistogramAxis

    **Attributes**

    .. autosummary::
        bincenters
        binwidths
        edges
        label
        limits
        max
        min
        nbins
        overflow

    **Methods**

    .. autosummary::
        __eq__
        __str__
        bin
        binwidth
        clone
        cut
        edge_index
        inaxis
        isuniform
        mergebins

Edges and Bins
--------------

.. autoattribute:: HistogramAxis.edges
    :annotation:
.. autoattribute:: HistogramAxis.min
.. autoattribute:: HistogramAxis.max
.. autoattribute:: HistogramAxis.limits
.. autoattribute:: HistogramAxis.overflow
.. automethod:: HistogramAxis.edge_index
.. automethod:: HistogramAxis.inaxis
.. automethod:: HistogramAxis.bin
.. autoattribute:: HistogramAxis.nbins
.. automethod:: HistogramAxis.binwidth
.. autoattribute:: HistogramAxis.binwidths
.. automethod:: HistogramAxis.isuniform

    The axis is considered uniform if the maximum deviation of the bin-widths is less than the given tolerance:

    .. math::

        \max\left(\frac{\left|w_{i} - \left<w\right>\right|}{\left<w\right>}\right) < tol

    where :math:`\left<w\right>` is the :py:func:`median <numpy.median>` bin width and :math:`max()` is evaluated over all bin widths, :math:`w_{i}`.

.. autoattribute:: HistogramAxis.bincenters

Tranformations
--------------

.. automethod:: HistogramAxis.cut
.. automethod:: HistogramAxis.mergebins

Misc
----

.. autoattribute:: HistogramAxis.label
    :annotation:
.. automethod:: HistogramAxis.clone
.. automethod:: HistogramAxis.__eq__

    Each point in the edge arrays must satisfy

    .. math::

        \left(\frac{\left|e - e^{\prime}\right|}{0.5 (e + e^{\prime})}\right) < tol,

    where

    .. math::

        e := \mathtt{self.edges[i]}

    and

    .. math::

        e^{\prime} := \mathtt{that.edges[i]}.

.. automethod:: HistogramAxis.__str__

