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
        mid
        min
        nbins
        overflow_value
        range

    **Methods**

    .. autosummary::
        __eq__
        __str__
        asdict
        bin
        binwidth
        copy
        cut
        edge_index
        fromdict
        inaxis
        isidentical
        isuniform
        mergebins

Edges
-----

.. autoattribute:: HistogramAxis.edges
    :annotation:
.. autoattribute:: HistogramAxis.min
.. autoattribute:: HistogramAxis.mid
.. autoattribute:: HistogramAxis.max
.. autoattribute:: HistogramAxis.limits
.. autoattribute:: HistogramAxis.range
.. automethod:: HistogramAxis.edge_index

Bins
----

.. automethod:: HistogramAxis.bin
.. automethod:: HistogramAxis.inaxis
.. autoattribute:: HistogramAxis.overflow_value
.. autoattribute:: HistogramAxis.nbins
.. automethod:: HistogramAxis.binwidth
.. autoattribute:: HistogramAxis.binwidths
.. automethod:: HistogramAxis.isuniform
.. autoattribute:: HistogramAxis.bincenters

Tranformations
--------------

.. automethod:: HistogramAxis.cut
.. automethod:: HistogramAxis.mergebins

Misc
----

.. autoattribute:: HistogramAxis.label
    :annotation:
.. automethod:: HistogramAxis.copy
.. automethod:: HistogramAxis.asdict
.. automethod:: HistogramAxis.fromdict
.. automethod:: HistogramAxis.isidentical
.. automethod:: HistogramAxis.__eq__
.. automethod:: HistogramAxis.__str__
