# coding: utf-8
from builtins import str

from copy import copy, deepcopy
from collections import Iterable, namedtuple
import numpy as np

from .detail import string_types

class HistogramAxis(object):
    r"""A single axis used internally by :class:`Histogram` to store
    the bin edges in an open-grid format.

    An axis consists of a continuous range on the real-line, divided
    into bins such that there are no breaks. The bins do not have to be
    uniform, but they do have to cover the entire range.

    Args:
        bins (int, float array): Number of bins in combination with
        `limits`, or 1D array of edges including the upper limit of the
        last bin.
        limits (float 2-tuple): `(min,max)` representing the limits of
        this axis used in combination with integer `bins`.
        label (str): The axis label including units if applicable.

    Raises:
        TypeError:  If `bins` is a list and `limits` is not None.
        ValueError: If arguments could not be understood as an axis
                    along the real line.

    Examples:
        The following examples both create an axis of 100 uniform bins
        from 0 to 10. In this case, there are 101 edges to this axis::

            a1 = HistogramAxis(100,[0,10])
            a2 = HistogramAxis(np.linspace(0,10,101))
            assert a1 == a2

        A non-uniform binning example: 10 bins on a log-scale from 1 to
        10000::

            a3 = HistogramAxis(np.logspace(0,4,11))

    """
    def __init__(self, bins, limits=None, label=None):
        # allow second argument to be the label if it is a string,
        # bins is iterable and no other argument is used
        if (isinstance(bins, Iterable) and
                label is None and
                isinstance(limits, string_types)):
            label = limits
            limits = None

        if __debug__:
            if limits is None:
                if not isinstance(bins, Iterable):
                    raise ValueError('bins must be iterable length >= 2')
            else:
                if len(limits) != 2:
                    raise ValueError('limits must be iterable length 2')
                if bins < 1:
                    raise ValueError('must have at least one bin')
            if not (isinstance(label, string_types) or (label is None)):
                raise ValueError('label must be string or None')

        if limits is None:
            self.edges = np.asarray(bins)
        else:
            self.edges = np.linspace(limits[0], limits[1], bins + 1)

        if label is not None:
            self.label = label

    def __str__(self):
        """String representation the edges array.

        The attribute :py:attr:`HistogramAxis.label` is not included in
        the output.
        """
        return str(self.edges)

    def __repr__(self):
        """Complete string representation of the histogram axis."""
        fmt = 'HistogramAxis(bins={}{})'
        if self.label is None:
            lbl = ''
        else:
            lbl = ', label="{}"'.format(self.label)
        return fmt.format(repr(self.edges.tolist()), lbl)

    def __eq__(self, that):
        r"""Compare edges to within numpy's default tolerance.
        Labels are ignored."""
        try:
            return np.allclose(self.edges,that.edges)
        except ValueError:
            return False

    def isidentical(self, that):
        r"""Deep comparison of this axis with another."""
        if not (self == that):
            return False
        if self.label != that.label:
            return False
        return True

    @property
    def edges(self):
        """The list of bin edges from low to high."""
        return self._edges

    @edges.setter
    def edges(self,e):
        if not isinstance(e,np.ndarray):
            e = np.asarray(e)
        if __debug__:
            if len(e) < 2:
                raise ValueError('edges must have two or more values')
            if e.ndim != 1:
                raise ValueError('edges must be a one-dimensional array')
            if not all(x < y for x, y in zip(e, e[1:])):
                raise ValueError('edges must be strictly increasing')
        self._edges = e

    @property
    def label(self):
        """The label of this axis including units if applicable.
        Example: "distance (meters)"."""
        return getattr(self, '_label', None)

    @label.setter
    def label(self, l):
        self._label = str(l)

    @label.deleter
    def label(self):
        del self._label

    def asdict(self):
        ret = {'edges':self.edges}
        if self.label is not None:
            ret.update(label=self.label)
        return ret

    @staticmethod
    def fromdict(d):
        label = d.get('label', None)
        if label is not None:
            label = str(label)
        return HistogramAxis(bins=d['edges'], label=label)

    @property
    def nbins(self):
        """Number of bins in this axis."""
        return len(self.edges) - 1

    @property
    def min(self):
        """Value of the lowest edge."""
        return self.edges[0]

    @property
    def max(self):
        """Value of the highest edge."""
        return self.edges[-1]

    @property
    def mid(self):
        """Value of middle of this axis."""
        return 0.5 * (self.min + self.max)

    @property
    def limits(self):
        """2-tuple of the lowest and highest edges."""
        return (self.min, self.max)

    @property
    def range(self):
        """Total range of this axis, min to max."""
        return self.max - self.min

    def binwidths(self):
        """All bin widths as an array."""
        return self.edges[1:] - self.edges[:-1]

    def bincenters(self):
        """Centers of all bins as an array."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def overflow_value(self):
        """Value guaranteed to be outside the range of this axis."""
        return self.max + 1.

    def __deepcopy__(self, memo):
        """Deep copy of this instance."""
        cls = self.__class__
        newaxis = cls.__new__(cls)
        memo[id(self)] = newaxis
        newaxis._edges = deepcopy(self.edges, memo)
        newaxis._label = deepcopy(self.label, memo)
        return newaxis

    def __copy__(self):
        """Deep copy of this instance."""
        return deepcopy(self)

    def copy(self):
        """Return a copy of this axis."""
        return deepcopy(self)

    def inaxis(self, x):
        """Check if `x` is within this axis.

        Arguments:
            x (float): Position along this axis.

        Returns:
            bool: Result of ``min <= x < max``.

        This follows the edge rules of :py:func:`numpy.digitize`:
        ``min <= x < max``.
        """
        return self.edges[0] <= x < self.edges[-1]

    def bin(self, x):
        """Bin index for the value `x`.

        Arguments:
            x (float or array of floats): Position(s) along this axis.

        Returns:
            int or array of ints: Bin(s) corresponding to the value `x`.

        Notes:
            This follows the convention: low <= x < high.
        """
        return np.searchsorted(self.edges, x, side='right') - 1

    def edge_index(self, x, snap='nearest'):
        """Index of the edge based on the given snap.

        Will always return a valid index from 0 to ``len(self.edges)-1``.

        Arguments:
            x (float): Position along this axis.
            snap (str): Method for determining which edge is returned.
            Possible values are: {'nearest','low','high','both'}.

        Note:
            Output is a single number except for when ``snap='both'``
            where the output will be a tuple: `(low,high)`.
        """

        snap_options = ['nearest','low','high','both']

        if __debug__:
            if snap not in snap_options:
                raise ValueError('Unknown snap keyword: ' + snap)

        minbin = 0
        maxbin = len(self.edges)-1

        bin = self.bin(x)
        lowbin = max(bin,minbin)
        highbin = min(bin+1,maxbin)

        if snap == 'both':
            return (lowbin,highbin)
        elif snap == 'high':
            return highbin
        elif snap == 'low':
            return lowbin
        else:  # snap == 'nearest'
            low,high = self.edges[[lowbin,highbin]]
            dlow = abs(x-low)
            dhigh = abs(high-x)
            if dlow < dhigh:
                return lowbin
            else:
                return highbin

    def binwidth(self, b=1):
        """Width of a specific bin.

        Arguments:
            b (int): bin index

        Returns:
            float: Width of the (b+1)-th bin.

        Bin indexing starts with zero and by default, the width of the
        second bin (index: 1) is returned.
        """
        return (self.edges[b + 1] - self.edges[b])

    def isuniform(self, rtol=1e-05, atol=1e-08):
        """Check if all bins are of equal width.

        Arguments:
            rtol (float): relative tolerance parameter
            atol (float): absolute tolerance parameter

        Returns:
            bool: True if all bins are the same size

        Note:
            This returns True if all bin-widths satisfy the following::

                abs(widths - mean) <= (atol + rtol * abs(median))

        """
        widths = self.binwidths()
        median = np.median(widths)
        return np.allclose(widths, median, rtol=rtol, atol=atol)

    _CutResult = namedtuple('CutResult', ['axis', 'mask'])

    def cut(self, low, high=None, snap='nearest'):
        """Return a truncated :py:class:`HistogramAxis`

        Arguments:
            low (float): Lowest bin will include this point
                    (`None` is equivalent to the lowest edge).
            high (float): Highest bin will include this point (`None`
            is equivalent to the highest edge).
            snap (str, str 2-tuple): Clip edges when creating new axis.
            Possible values are:
            {'nearest','expand','low','high','clip'}.

        Returns:
            :py:class:`HistogramAxis`: A new instance.
            mask (boolean array): mask array to be used on data to make
            this cut

        Note:
            Full bins of this axis will be used unless
            ``snap == 'clip'`` in which case the low or high bins will
            be clipped. Clipping will likely make a uniform axis no
            longer uniform.
        """
        snap_options = ['nearest', 'expand', 'low', 'high', 'clip']

        if isinstance(snap, string_types):
            snap = (snap, snap)

        if __debug__:
            for s in snap:
                if s not in snap_options:
                    raise ValueError('Unknown snap keyword: ' + s)

        if low is None:
            lowi = 0
        else:
            if snap[0] in ['clip', 'expand']:
                eisnap = 'low'
            else:
                eisnap = snap[0]
            lowi = self.edge_index(low, eisnap)

            if eisnap in ['high']:
                if lowi > 0 and np.isclose(self.edges[lowi - 1], low):
                    lowi -= 1

        if high is None:
            highi = len(self.edges) - 1
        else:
            if snap[1] in ['clip', 'expand']:
                eisnap = 'high'
            else:
                eisnap = snap[1]
            highi = self.edge_index(high, eisnap)

            if eisnap == 'high':
                if highi > 0 and np.isclose(self.edges[highi - 1], high):
                    highi -= 1

        newedges = np.copy(self.edges[lowi:highi + 1])
        mask = np.zeros((self.nbins,), dtype=np.bool)
        mask[lowi:highi] = True

        if low is not None:
            if snap[0] == 'clip':
                newedges[0] = max(low, self.edges[lowi])
        if high is not None:
            if snap[1] == 'clip':
                newedges[-1] = min(high, self.edges[highi])

        newaxis = HistogramAxis(newedges, label=copy(self.label))
        return HistogramAxis._CutResult(newaxis, mask)

    def mergebins(self, nbins=2, snap='low', clip=True):
        """Merge neighboring bins.

        Arguments:
            nbins (int): Number of bins to merge.
            snap (str): If `nbins` does not evenly divide the number
                    of bins in this axis, this will "snap" the window
                    to the "high" or "low" edge.
            clip (bool): Ignore opposite limit from `snap` if `nbins`
                    does not evenly divide the number of bins in this
                    axis. Setting this to True (default) will result
                    in an axis that maintains uniform bin widths.

        Returns:
            :py:class:`HistogramAxis`: A new instance.
        """
        snap_options = ['low','high']

        if __debug__:
            if snap not in snap_options:
                raise ValueError('Unknown snap keyword: ' + snap)

        d, m = divmod(self.nbins, nbins)

        if m == 0:
            newedges = self.edges[::nbins]
        elif clip:
            if snap == 'low':
                newedges = self.edges[::nbins]
            else:
                newedges = self.edges[m::nbins]
        else:
            if snap == 'low':
                newedges = np.concatenate((self.edges[::nbins],
                                           [self.edges[-1]]))
            else:
                newedges = np.concatenate(([self.edges[0]],
                                           self.edges[m::nbins]))

        return HistogramAxis(newedges, label=copy(self.label))
