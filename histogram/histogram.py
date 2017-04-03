from __future__ import division, unicode_literals
from builtins import str
from six import string_types, text_type

import itertools as it

from collections import Iterable
from copy import copy, deepcopy
from numbers import Integral
from warnings import warn

import numpy as np
from scipy import optimize as opt
from scipy import stats, ndimage, interpolate
from uncertainties import nominal_value, std_dev, ufloat
from uncertainties import unumpy as unp

from .histogram_axis import HistogramAxis
from .detail import skippable, window
from . import rc

# ignore divide by zero (silently create nan's)
np.seterr(divide='ignore', invalid='ignore')

class Histogram(object):
    """N-dimensional histogram over a continuous range.

    This is a histogram where each axis is a continuous (non-discrete) range
    with a set number of bins. The binning does not have to be evenly spaced.

    Args:
        axes (list): List of :py:class:`HistogramAxis` or constructor
            parameters thereof. These are the axis definitions.

    Keyword Args:
        label (str): Label for the filled data.
        title (str): Title of this histogram.
        data (scalar array): N-dimensional array for the filled data.
        uncert (scalar array): N-dimensional array for the uncertainty.
        dtype (scalar type): Type of the data array. Input data will be
            converted if different.

    Example:

        Typical usage would be to fill the histogram from a sample of data. In
        this example, we create a 1D histogram with 100 bins from 0 to 10, and
        fill it with 10k samples distributed normally around 5 with a width
        (sigma) of 1::

            import numpy as np
            from matplotlib import pyplot
            from histogram import Histogram

            h = Histogram(100, [0, 10], 'x (cm)', 'counts', 'Random Distribution')
            h.fill(np.random.normal(5, 1, 10000))
            fig, ax = pyplot.subplots(figsize=(4, 2.5))
            fig.subplots_adjust(left=.18, bottom=.2, right=.95, top=.88)
            pt = ax.plothist(h, color='steelblue')
            pyplot.show()

        .. image:: images/histogram_1dnorm.png
    """
    def __init__(self, *axes, **kwargs):
        label  = kwargs.pop('label' , None)
        title  = kwargs.pop('title' , None)
        data   = kwargs.pop('data'  , None)
        dtype  = kwargs.pop('dtype' , None)
        uncert = kwargs.pop('uncert', None)

        if not axes:
            raise TypeError('you must specify at least one axis.')

        self.axes = []
        for skip, (arg0, arg1, arg2) in skippable(window(axes, size=3)):
            if (isinstance(arg0, Iterable) and
                    not isinstance(arg0, string_types)):
                try:
                    arg0_array = np.asarray(arg0)
                    if (arg0_array.dtype == object) or (len(arg0_array.shape) != 1):
                        self.axes.append(HistogramAxis(*arg0))
                    elif isinstance(arg1, string_types):
                        self.axes.append(HistogramAxis(arg0_array, arg1))
                        skip(1)
                    else:
                        self.axes.append(HistogramAxis(arg0_array))
                except ValueError:
                    self.axes.append(HistogramAxis(*arg0))
            elif isinstance(arg0, Integral):
                if isinstance(arg2, string_types):
                    self.axes.append(HistogramAxis(arg0, arg1, arg2))
                    skip(2)
                else:
                    self.axes.append(HistogramAxis(arg0, arg1))
                    skip(1)
            elif isinstance(arg0, HistogramAxis):
                self.axes.append(arg0)
            else:
                assert isinstance(arg0, string_types) or arg0 is None
                assert isinstance(arg1, string_types) or arg1 is None
                assert arg2 is None
                for a in (arg0, arg1):
                    if isinstance(a, string_types):
                        if label is None:
                            label = a
                        elif title is None:
                            title = a
                        else:
                            raise TypeError('bad argument list')
                skip(1)

        self.label = label
        self.title = title

        shape = tuple(ax.nbins for ax in self.axes)

        if data is None:
            self._data = np.zeros(shape=shape, dtype=(dtype or rc.fill_type))
        else:
            self._data = np.asarray(data)
            if dtype is not None:
                self._data = self._data.astype(dtype)
            assert self._data.shape == shape, 'Data shape must match axes.'

        if uncert is not None:
            self.uncert = uncert

### properties
    @property
    def data(self):
        """:py:class:`numpy.ndarray` of the filled data.

        The indexes are in the same order as the :py:class:`HistogramAxis`
        objects in the list stored in :py:attr:`Histogram.axes`. One can set
        this directly - shape is checked and data is written "in-place" when
        possible.

        Here, we create a histogram and set the data directly::

            from scipy import stats
            from matplotlib import pyplot
            from histogram import Histogram

            h = Histogram(50, [0, 10])

            xx, = h.grid
            h.data = 1000 * stats.norm(5, 2).pdf(xx)

            fig, ax = pyplot.subplots(figsize=(4, 2.5))
            fig.subplots_adjust(left=.15, bottom=.2, right=.9, top=.85)
            pt = ax.plothist(h, color='steelblue')
            pyplot.show()

        .. image:: images/histogram_data_1dnorm.png
        """
        return self._data

    @data.setter
    def data(self, d):
        self._data[...] = d

    @property
    def has_uncert(self):
        return hasattr(self, '_uncert')

    @property
    def uncert(self):
        """:py:class:`numpy.ndarray` of the absolute uncertainty.

        This has the same shape as :py:attr:`Histogram.data`. Under certain
        cases, this will be set automatically to the square-root of the data
        (Poisson statistics assumption).

        When histogramming a sample of randomly distributed data, the
        uncertainty of each bin is the equal to the square-root of the counts
        in that bin::

            import numpy as np
            from scipy import stats
            from numpy import random as rand
            from matplotlib import pyplot
            from histogram import Histogram

            rand.seed(1)

            h = Histogram(30, [0, 10])

            xx, = h.grid
            h.data = 1000 * stats.norm(5, 2).pdf(xx)
            h.data += rand.normal(0, 10, xx.shape)

            h.uncert = np.sqrt(h.data)

            fig, ax = pyplot.subplots(figsize=(4, 2.5))
            fig.subplots_adjust(left=.15, bottom=.2, right=.9, top=.85)
            pt = ax.plothist(h, style='errorbar')
            pyplot.show()

        .. image:: images/histogram_uncert_1dnorm.png
        """
        return getattr(self, '_uncert', np.sqrt(self.data))

    @uncert.setter
    def uncert(self, u):
        if u is None:
            del self.uncert
        else:
            if not self.has_uncert:
                self._uncert = np.empty(self.data.shape, dtype=np.float64)
            self._uncert[...] = u

    @uncert.deleter
    def uncert(self):
        if self.has_uncert:
            del self._uncert

    @property
    def uncert_ratio(self):
        """The untertainty as ratios of the data"""
        return self.uncert / self.data

    @uncert_ratio.setter
    def uncert_ratio(self, u):
        """Set the uncertainty by ratio of the data"""
        self.uncert = u * self.data

    @property
    def title(self):
        """Title of this histogram."""
        return getattr(self, '_title', None)

    @title.setter
    def title(self, t):
        if t is None:
            del self.title
        else:
            self._title = text_type(t)

    @title.deleter
    def title(self):
        if hasattr(self, '_title'):
            del self._title

    @property
    def label(self):
        """Label of the filled data in this histogram

        This is usually something like "counts"."""
        return getattr(self, '_label', None)

    @label.setter
    def label(self, l):
        if l is None:
            del self.label
        else:
            self._label = text_type(l)

    @label.deleter
    def label(self):
        if hasattr(self, '_label'):
            del self._label

    def __eq__(self, that):
        """Check if data and axes are equal.

        Uncertainty, labels and titles are not considered. For complete
        equality, see :py:meth:`Histogram.isidentical`. For finer control of
        histogram comparison, consider using :py:func:`numpy.allclose()` on the
        data::

            import numpy as np
            from histogram import Histogram

            h1 = Histogram(10, (0, 10))
            h2 = Histogram(10, (0, 10))
            h3 = Histogram(10, (-10, 10))

            h1.data[2] = 1
            h2.data[2] = 1
            h3.data[3] = 1

            # checks data and axes:
            assert h1 == h2
            assert not (h1 == h3)

            # checks only data:
            assert np.allclose(h1.data, h2.data)
        """
        try:
            if not np.allclose(self.data, that.data):
                return False
            for a, aa in zip(self.axes, that.axes):
                if not (a == aa):
                    return False
        except ValueError:
            # histogram data shape mismatch
            return False
        return True

    def __ne__(self, that):
        return not (self == that)

    def isidentical(self, that):
        """Check if histograms are identical including uncertainty and labels.

        See also :py:meth:`Histogram.__eq__`."""
        if not (self == that):
            return False
        if self.has_uncert or that.has_uncert:
            if not np.allclose(self.uncert, that.uncert):
                return False
        if self.label != that.label:
            return False
        if self.title != that.title:
            return False
        for a, aa in zip(self.axes, that.axes):
            if not a.isidentical(aa):
                return False
        return True

### non-modifying information getters
    def __str__(self):
        """Breif string representation of the data.

        Returns the string representation of the numpy array containing the
        data only. Axes, uncertainty and labels are ignored.

        Example::

            from numpy import random as rand
            from histogram import Histogram

            rand.seed(1)

            h = Histogram(10, [0, 10])
            h.fill(rand.normal(5, 2, 10000))
            print(h)

        output::

            [ 164  428  909 1484 1915 1934 1525  873  467  175]

        """
        return str(self.data)

    def __repr__(self):
        """Complete string representation of the histogram"""
        fmt = 'Histogram({axes}, {args})'
        axesstr = ', '.join(repr(a) for a in self.axes)
        args = {
            'data': repr(self.data.tolist()),
            'dtype':'"{}"'.format(str(self.data.dtype)) }
        if self.label is not None:
            args['label'] = '"{}"'.format(self.label)
        if self.title is not None:
            args['title'] = '"{}"'.format(self.title)
        if self.has_uncert:
            args['uncert'] = str(self.uncert.tolist())
        argsstr = ', '.join('{}={}'.format(k, v)
                            for k, v in sorted(args.items()))
        return fmt.format(axes=axesstr, args=argsstr)

    def __call__(self, *xx, **kwargs):
        """Value of histogram at a point

        Returns the value of the histogram at a specific point ``(x, y...)`` or
        array of points ``(xx, yy...)``.

        Args:
            xx (tuple of numbers or arrays): Point(s) inside the axes of this
                histogram.

        Keyword Args:
            overflow_value (number): Return value when the point lies outside
                this histogram. (default: 0)

        Example::

            from numpy import random as rand
            from histogram import Histogram

            rand.seed(1)

            h = Histogram(10, [0, 10])
            h.fill(rand.normal(5, 2, 10000))
            for x in [0.5, 1.5, 2.5]:
                print( h(x) )

        output::

            164
            428str(
            909

        """
        overflow_value = kwargs.pop('overflow_value', 0)

        bin = []
        for x, ax in zip(xx, self.axes):
            b = ax.bin(x)
            if (b < 0) or (b >= ax.nbins):
                return overflow_value
            bin += [b]

        return self.data[tuple(bin)]

    def asdict(self, encoding=None, flat=False):
        """Dictionary representation of this histogram.

        This includes uncertainty, axes, labels and title and is used to
        serialize the histogram to NumPy's binary format (see
        :py:func:`save_histogram_to_npz`).

        """
        ret = {'data' : self.data}
        if flat:
            for i, ax in enumerate(self.axes):
                for k, v in ax.asdict(encoding).items():
                    key = 'axes:{}:{}'.format(i, k)
                    ret[key] = v
        else:
            ret['axes'] = [a.asdict(encoding) for a in self.axes]
        if self.has_uncert:
            ret['uncert'] = self.uncert
        if self.label is not None:
            if encoding is not None:
                ret['label'] = self.label.encode(encoding)
            else:
                ret['label'] = self.label
        if self.title is not None:
            if encoding is not None:
                ret['title'] = self.title.encode(encoding)
            else:
                ret['title'] = self.title
        return ret

    @staticmethod
    def fromdict(d, encoding=None):
        """Create new :py:class:`Histogram` from a dictionary."""
        if 'axes' in d:
            axes = [HistogramAxis.fromdict(a, encoding) for a in d.pop('axes')]
        else:
            axes = []
            for i in range(d['data'].ndim):
                axdict = {}
                for k in ['edges', 'label']:
                    key = 'axes:{}:{}'.format(i, k)
                    if key in d:
                        axdict[k] = d.pop(key)
                axes.append(axdict)
            axes = [HistogramAxis.fromdict(a, encoding) for a in axes]
        if encoding is not None:
            if 'label' in d:
                d['label'] = d['label'].decode(encoding)
            if 'title' in d:
                d['title'] = d['title'].decode(encoding)
        return Histogram(*axes, **d)

###    dimension and shape
    @property
    def dim(self):
        """Dimension of this histogram (number of axes)"""
        return len(self.axes)

    @property
    def shape(self):
        """Shape of the histogram data

        This is a tuple of the number of bins in each axis ``(x, y...)``. This
        is the same as :py:attr:`Histogram.data.shape`.

        """
        return self.data.shape

    @property
    def size(self):
        """Total number of bins (size of data)

        This is the product of the number of bins in each axis. This is the
        same as :py:meth:`Histogram.data.size`.

        """
        return self.data.size

###    axes information (bin edges and centers)
    def isuniform(self, rtol=1e-05, atol=1e-08):
        """Check if all axes are uniform

        Returns "and" of :py:meth:`HistogramAxis.isuniform` for each axis.
        """
        return all([ax.isuniform(rtol=rtol, atol=atol) for ax in self.axes])

    @property
    def edges(self):
        """Edges of each axis as a tuple

        Output is in the form::

            ( [x0, x1..], [y0, y1...] ... )
        """
        return tuple([ax.edges for ax in self.axes])

    def grid(self):
        """Meshgrid of the.bincenters() of each axis

        This is a single array for 1D histograms - i.e. the bin-centers of the
        x-axis. For 2D histograms, this is a tuple of two 2D arrays::

            XX, YY = h2.grid

        Here, ``XX`` and ``YY`` are arrays of shape ``(xbins, ybins)``. For 1D
        histograms, the output is still a tuple so typically, one should expand
        this out with a comma::

            xx, = h1.grid
        """
        if self.dim == 1:
            return (self.axes[0].bincenters(), )
        else:
            centers = [ax.bincenters() for ax in self.axes]
            return np.meshgrid(*centers, indexing='ij')

    def edge_grid(self):
        """Meshgrid built from the axes' edges

        This is the same as :py:meth:`Histogram.grid` but for the edges of each
        axis instead of the bin centers.

        """
        if self.dim == 1:
            return (self.axes[0].edges, )
        else:
            edges = [ax.edges for ax in self.axes]
            return np.meshgrid(*edges, indexing='ij')

    def binwidths(self):
        """Widths of all bins along each axis

        This will always return a tuple::

            dx, dy = h2.binwidths()

        Here, ``dx`` and ``dy`` are arrays of the widths of each bin along the
        `x` and `y` axes respecitively. For 1D histograms, the output is still
        a tuple so typically, one should expand this out with a comma::

            dx, = h1.binwidths()
        """
        return tuple([ax.binwidths() for ax in self.axes])

    def binwidth(self, b=1, axis=0):
        """Width of a specific bin ``b`` along an axis

        Args:
            b (int): Bin index (from zero, default: 1).
            axis (int): Axis index (default: 0).

        Note:

            Default is the second bin (index = 1) in the first (index = 0) axis.
        """
        return self.axes[axis].binwidth(b)

    def binvolumes(self):
        """Volumes of each bin

        Volume is defined as the product of the bin-widths along each axis for the given bin. For 1D histogram, this is the same as :py:attr:`Histogram.binwidths()`. For 2D histograms, this returns a 2D array like the following where dxi is the width of the ith bin along the x-axis (first, index = 0)::

            [ [ dx0*dy0, dx0*dy1 ... ],
              [ dx1*dy0, dx1*dy1 ... ],
              ... ] = h2.binvolumes()

            h.binvolumes()[i, j] = dxi * dyj
        """
        widths = self.binwidths()
        if self.dim == 1:
            return widths
        else:
            return np.multiply.reduce(np.ix_(*widths))

    @property
    def overflow_value(self):
        """Guaranteed overflow point when filling this histogram

        For 1D histograms, this is a tuple of one value ``(x, )`` generated by
        :py:attr:`HistogramAxis.overflow_value`. For 2D histograms, this will
        look like ``(x, y)``.

        Example::

            from histogram import Histogram

            ha = Histogram(10, [0, 10])
            print(h)

            ha.fill(ha.overflow_value)
            print(h)

        Output::

            [0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0]
        """
        return tuple(ax.overflow_value for ax in self.axes)

###    data information (limits, sum, extent)
    def sum_data(self, *axes):
        """Sum of bin values or sum along one or more axes."""
        all_axes = tuple(range(self.dim))
        axes = all_axes if len(axes) == 0 else tuple(sorted(axes))
        if not self.has_uncert and axes == all_axes:
            s = self.data.sum()
            result = ufloat(s, np.sqrt(s))
        else:
            result = np.sum(unp.uarray(self.data, self.uncert), axis=axes)
        return result

    def sum(self, *axes):
        """Sum of bin values or sum along one or more axes.

        Args:

            axes (tuple of integers, optional): Axes to sum over.

        Returns the sum over all values or sums only over specific axes and
        returns a new :py:class:`Histogram` with reduced dimension.

        Example::

            from histogram import Histogram

            h = Histogram(10, [0, 10])
            h.fill([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

            print('sum of h:', h.sum())

            h2 = Histogram(10, [0, 10], 10, [0, 10])
            h2.fill([1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
                    [1, 1, 1, 1, 1, 1, 2, 2, 2, 2])

            print('h2 sum along axis 0:', h2.sum(0))

        Output::

            sum of h: 10
            h2 sum along axis 0: [0 6 4 0 0 0 0 0 0 0]
        """
        all_axes = tuple(range(self.dim))
        axes = all_axes if len(axes) == 0 else tuple(sorted(axes))
        if axes == all_axes:
            return self.sum_data()
        else:
            if self.has_uncert:
                result = self.sum_data(*axes)
                newdata = unp.nominal_values(result)
                newuncert = unp.std_devs(result)
            else:
                newdata = np.sum(self.data, axis=axes)
                newuncert = None
            ii = sorted(set(range(self.dim)) - set(axes))
            newaxes = [self.axes[i] for i in ii]
            return Histogram(*newaxes, data=newdata, uncert=newuncert,
                             title=copy(self.title), label=copy(self.label))

    def projection_data(self, axis):
        """Projection of the data onto an axis."""
        sumaxes = set(range(self.dim)) - {axis}
        return self.sum_data(*sumaxes)

    def projection(self, axis):
        """Projection onto a single axis."""
        sumaxes = set(range(self.dim)) - {axis}
        return self.sum(*sumaxes)

    def integral(self):
        """Total volume-weighted sum of the histogram."""
        res = np.sum(unp.uarray(self.data, self.uncert) * self.binvolumes())
        return nominal_value(res), std_dev(res)

    def min(self):
        """Minimum value of the filled data including uncertainty."""
        return np.nanmin(self.data - self.uncert)

    def max(self):
        """Maximum value of the filled data including uncertainty."""
        return np.nanmax(self.data + self.uncert)

    def mean(self):
        """Mean position of the data along the axes

        Returns:

            tuple: Mean (float) along each axis: ``(xmean, ymean...)``.

        Bin-centers are used as the position and non-equal widths are
        incorporated into the weighting of the results.

        """
        mean = []
        for i, axis in enumerate(self.axes):
            if self.dim > 1:
                w = self.projection_data(i)
            else:
                w = unp.uarray(self.data, self.uncert)
            if axis.isuniform():
                x = unp.uarray(axis.bincenters(), 0.5 * axis.binwidth())
            else:
                bw = axis.binwidths()
                x = unp.uarray(axis.bincenters(), 0.5 * bw)
                w *= bw
            mean.append(np.sum(x * w) / np.sum(w))
        return tuple(mean)

    def var(self):
        """Variance of the data along the axes

        Returns:

            tuple: Variance (float) along each axis: ``(xvar, yvar...)``.

        This will ignore all ``nan`` and ``inf`` values in the histogram.
        Bin-centers are used as the position and non-equal widths are
        incorporated into the weighting of the results.

        """
        var = []
        for i, (axis, mean) in enumerate(zip(self.axes, self.mean())):
            if self.dim > 1:
                w = self.projection_data(i)
            else:
                w = unp.uarray(self.data, self.uncert)
            if axis.isuniform():
                x = unp.uarray(axis.bincenters(), 0.5 * axis.binwidth())
            else:
                bw = axis.binwidths()
                x = unp.uarray(axis.bincenters(), 0.5 * bw)
                w *= bw
            sum_w = np.sum(w)
            mean = np.sum(w * x) / sum_w
            var.append(np.sum(w * (x - mean)**2) / sum_w)
        return tuple(var)

    def std(self):
        """Standard deviation of the data along the axes

        Returns:

            tuple: Standard deviation (float) along each axis: ``(xstd,
            ystd...)``.
        """
        var = [ufloat(x.n,x.s) for x in self.var()]
        return tuple(unp.sqrt(var))

    def extent(self, maxdim=None, uncert=True, pad=None):
        """Extent of axes and data

        Returns:

            tuple: extent like ``(xmin, xmax, ymin ...)``.

        By default, this includes the uncertainty if the last dimension is the
        histogram's data and not an axis::

            [xmin, xmax, ymin, ymax, ..., min(data-uncert), max(data+uncert)]

        padding is given as a percent of the actual extent of the axes or data
        (plus uncertainty) and can be either a single floating point number or
        a list of length ``2 * maxdim``.

        """
        if maxdim is None:
            maxdim = self.dim + 1
        ext = []
        for ax in self.axes[:maxdim]:
            ext += [ax.min, ax.max]
        if len(ext) < (2*maxdim):
            if uncert:
                ext += [self.min(), self.max()]
            else:
                ext += [np.nanmin(self.data), np.nanmax(self.data)]
        if pad is not None:
            if not isinstance(pad, Iterable):
                pad = [pad]*(2*maxdim)
            for dim in range(maxdim):
                a, b = 2*dim, 2*dim+1
                w = ext[b] - ext[a]
                ext[a] -= pad[a] * w
                ext[b] += pad[b] * w
        return tuple(ext)

    def errorbars(self, maxdim=None, asratio=False):
        """Bin half-widths and data uncertainties."""
        if maxdim is None:
            maxdim = self.dim + 1
        ret = [0.5 * ax.binwidths() for ax in self.axes[:maxdim]]
        if len(ret) < maxdim:
            ret += [self.uncert]

        if asratio:
            for x, ax in zip(ret, self.axes):
                x /= ax.range
            if maxdim > self.dim:
                ret[-1] /= self.data.max() - self.data.min()

        return ret

    def asline(self):
        """Points describing this histogram as a line."""
        assert self.dim == 1, 'only 1D histograms can be translated into a line.'

        x = self.axes[0].edges
        y = self.data

        xx = np.column_stack([x[:-1], x[1:]]).ravel()
        yy = np.column_stack([y, y]).ravel()

        extent = [min(xx), max(xx), min(yy), max(yy)]
        return xx, yy, extent

### self modifying methods (set, fill)
    def __getitem__(self, *args):
        """Direct access to the filled data."""
        return self.data.__getitem__(*args)

    def __setitem__(self, *args):
        """Direct access to the filled data."""
        return self.data.__setitem__(*args)

    def set(self, val, uncert=None):
        """Set filled data to specific values.

        This will set the uncertainty to ``None`` by default and will accept a
        single value or an array the same shape as the data. Data will be cast
        to the data type already stored in the histogram.
        """
        if isinstance(val, np.ndarray):
            self.data.T[...] = val.T
        else:
            self.data[...] = val

        if uncert is None:
            del self.uncert
        else:
            if not self.has_uncert:
                self._uncert = np.empty(self.data.shape)
            if isinstance(uncert, np.ndarray):
                self.uncert.T[...] = uncert.T
            else:
                self.uncert[...] = uncert

    def set_nans(self, val=0, uncert=0):
        """Set all NaNs to a specific value."""
        self.data[np.isnan(self.data)] = val
        if self.has_uncert:
            self.uncert[np.isnan(self.uncert)] = uncert

    def set_infs(self, val=0, uncert=0):
        """Set all infinity values to a specific value."""
        self.data[np.isinf(self.data)] = val
        if self.has_uncert:
            self.uncert[np.isinf(self.uncert)] = uncert

    def set_nonfinites(self, val=0, uncert=0):
        """Set all non-finite values to a specific value."""
        self.data[~np.isfinite(self.data)] = val
        if self.has_uncert:
            self.uncert[~np.isfinite(self.uncert)] = uncert

    def reset(self):
        """Set data to zero and uncertainty to `None`."""
        self.set(0)
        self.uncert = None

    def fill(self, *args):
        """Fill histogram with sample data.

        Arguments (``\*args``) are the sample of data with optional associated
        weights. Weights may be a single number or an array of length `N`. The
        default (``None``) is equivalent to ``weights=1``. Example::

            from histogram import Histogram

            ### 1D Example
            h = Histogram(10, [0, 10])

            h.fill(1)           # single value
            h.fill(2, 2)         # single value with weight
            h.fill([3, 3, 3])     # data sample
            h.fill([4, 4], 2)     # data sample with constant weight
            h.fill([5, 5], [2, 3]) # data sample with variable weights

            print(h)
            # output:
            # [0 1 2 3 4 5 0 0 0 0]

            ### 2D Example
            h = Histogram(3, [0, 3], 10, [0, 10])
            xdata = [0, 0, 1, 1, 2]
            ydata = [1, 2, 3, 4, 5]
            weights = [1, 2, 3, 4, 5]
            h.fill(xdata, ydata, weights)

            print(h)
            # output:
            # [[0 1 2 0 0 0 0 0 0 0]
            #  [0 0 0 3 4 0 0 0 0 0]
            #  [0 0 0 0 0 5 0 0 0 0]]
        """
        if len(args) > self.dim:
            sample = args[:-1]
            weights = args[-1]
        else:
            sample = args
            weights = None

        self.fill_from_sample(sample, weights)

    def fill_one(self, pt, wt=1):
        """Fill a single data point

        This increments a single bin by weight ``wt``. While it is the fastest
        method for a single entry, it should only be used as a last resort
        because its at least an order of magnitude slower than
        :py:meth:`Histogram.fill_from_sample` when filling many entries.

        """
        try:
            if pt < self.axes[0].min or self.axes[0].max < pt:
                return
            self.data[self.axes[0].bin(pt)] += wt
        except ValueError:
            b = []
            for x, ax, m in zip(pt, self.axes, self.data.shape):
                if x < ax.min or ax.max < x:
                    return
                b += [ax.bin(x)]
            self.data[tuple(b)] += wt

    def fill_from_sample(self, sample, weights=None):
        """Fill histogram from sample of data

        This fills the histogram from sample with shape `(D, N)` array where `D`
        is the dimension of the histogram and `N` is the number of points to
        fill. The optional ``weights`` may be a single number or an array of
        length `N`. The default (``None``) is equivalent to ``weights = 1``.

        This is the primary work-horse of the :py:class:`Histogram` class and
        should be favored, along with the wrapper method
        :py:meth:`Histogram.fill` over :py:meth:`Histogram.fill_one`.

        Example::

            from numpy import random as rand
            from matplotlib import pyplot
            from histogram import Histogram

            rand.seed(1)

            ### 1D Example
            h = Histogram(10, [0, 10], 30, [0, 10])

            h.fill_from_sample(rand.normal(5, 1, (2, 20000)))

            fig, ax = pyplot.subplots(figsize=(4, 2.5))
            pt = ax.plothist(h)
            fig.tight_layout()

            pyplot.show()

        .. image:: images/histogram_fill_from_sample.png
        """
        if not isinstance(sample, np.ndarray):
            sample = np.array(sample)
        wt = weights
        if weights is not None:
            if not isinstance(weights, Iterable):
                wt = np.empty((sample.T.shape[0], ))
                wt[...] = weights
        h, e = np.histogramdd(sample.T, self.edges, weights=wt)
        self.data += h.astype(self.data.dtype)

### operations
    def __deepcopy__(self, memo=None):
        """Create a complete copy of this histogram."""
        return self.copy()

    def __copy__(self):
        """Create a complete copy of this histogram."""
        return self.copy()

    def copy(self, dtype=None, **kwargs):
        """Copy this histogram optionally changing dtype and labels."""
        cls = self.__class__
        newhist = cls.__new__(cls)

        newhist.axes = [deepcopy(ax) for ax in self.axes]
        if dtype is None:
            newhist._data = deepcopy(self._data)
        else:
            newhist._data = self._data.astype(dtype)
        newhist.title = kwargs.get('title', deepcopy(self.title))
        newhist.label = kwargs.get('label', deepcopy(self.label))
        if self.has_uncert:
            newhist._uncert = copy(self._uncert)

        return newhist

    def __iadd__(self, that):
        """In-place addition."""
        if isinstance(that, Histogram):
            if self.has_uncert or that.has_uncert:
                self_data = unp.uarray(self.data, self.uncert)
                that_data = unp.uarray(that.data, that.uncert)
                self_data.T[...] += that_data.T
                self.data[...] = unp.nominal_values(self_data)
                self.uncert = unp.std_devs(self_data)
            else:
                self.data.T[...] += that.data.T
        else:
            self_data = unp.uarray(self.data, self.uncert)
            self_data.T[...] += np.asarray(that).T
            self.data[...] = unp.nominal_values(self_data)
            self.uncert = unp.std_devs(self_data)
        return self

    def __radd__(self, that):
        """Commuting addition."""
        return self + that

    def __add__(self, that):
        """Addition."""
        if isinstance(that, Histogram) and self.dim < that.dim:
            return that + self
        else:
            if isinstance(that, Histogram):
                that_dtype = that.data.dtype
            else:
                that_dtype = np.dtype(type(that))
            copy_dtype = None
            if self.has_uncert or (isinstance(that, Histogram) and
                                   that.has_uncert):
                copy_dtype = np.float64
            elif self.data.dtype != that_dtype:
                inttypes = [np.int32, np.int64]
                if (self.data.dtype in inttypes) and \
                   (that_dtype in inttypes):
                    copy_dtype = np.int64
                else:
                    copy_dtype = np.float64
            ret = self.copy(copy_dtype, label=None)
            ret += that
            return ret

    def __isub__(self, that):
        """In-place subtraction."""
        if isinstance(that, Histogram):
            self_data = unp.uarray(self.data, self.uncert)
            that_data = unp.uarray(that.data, that.uncert)
            self_data.T[...] -= that_data.T
            self.data[...] = unp.nominal_values(self_data)
            self.uncert = unp.std_devs(self_data)
        else:
            self_data = unp.uarray(self.data, self.uncert)
            self_data.T[...] -= np.asarray(that).T
            self.data[...] = unp.nominal_values(self_data)
            self.uncert = unp.std_devs(self_data)
        return self

    def __rsub__(self, that):
        """Commuting subtraction."""
        ret = self.copy(np.float64, label=None)
        ret.data.T[...] = unp.nominal_values(that).T
        ret._uncert = np.empty(shape=ret.data.shape)
        ret._uncert.T[...] = unp.std_devs(that).T
        ret -= self
        return ret

    def __sub__(self, that):
        """Subtraction."""
        ret = self.copy(np.float64, label=None)
        ret -= that
        return ret

    def __imul__(self, that):
        """In-place multiplication."""
        if isinstance(that, Histogram):
            self_data = unp.uarray(self.data, self.uncert)
            that_data = unp.uarray(that.data, that.uncert)
            self_data.T[...] *= that_data.T
            self.data[...] = unp.nominal_values(self_data)
            self.uncert = unp.std_devs(self_data)
        else:
            self_data = unp.uarray(self.data, self.uncert)
            self_data.T[...] *= np.asarray(that).T
            self.data[...] = unp.nominal_values(self_data)
            self.uncert = unp.std_devs(self_data)
        return self

    def __rmul__(self, that):
        """Commuting mulitplication."""
        return self * that

    def __mul__(self, that):
        """Multiplication."""
        ret = self.copy(np.float64, label=None)
        ret *= that
        return ret

    def __itruediv__(self, that):
        """In-place (true) division."""
        if isinstance(that, Histogram):
            infs = np.isclose(that.data, 0)
            nans = np.isclose(self.data, 0) & infs
            ninfs = (self.data < 0) & infs
            sel = ~(infs | nans)
            self_data = unp.uarray(self.data[sel], self.uncert[sel])
            that_data = unp.uarray(that.data[sel], that.uncert[sel])
            self_data.T[...] /= that_data.T
            self.data[sel] = unp.nominal_values(self_data)
            self.uncert[sel] = unp.std_devs(self_data)
            self.data[infs] = np.inf
            self.data[ninfs] = -np.inf
            self.data[nans] = np.nan
            self.uncert[infs | nans] = np.nan
        else:
            self_data = unp.uarray(self.data, self.uncert)
            self_data.T[...] /= np.asarray(that).T
            self.data[...] = unp.nominal_values(self_data)
            self.uncert = unp.std_devs(self_data)
        return self

    def __rtruediv__(self, that):
        """Commuting (true) division.

            that = 1.
            hret = that / hself
        """
        ret = self.copy(np.float64, label=None)
        ret.data.T[...] = unp.nominal_values(that).T
        ret._uncert = np.empty(shape=ret.data.shape)
        ret._uncert.T[...] = unp.std_devs(that).T
        ret /= self
        return ret

    def __truediv__(self, that):
        """Division."""
        ret = self.copy(np.float64)
        ret /= that
        return ret

### interpolating and smoothing
    def interpolate_nonfinites(self, method='cubic', **kwargs):
        """Replace non-finite bins with interpolated values.

        Keyword Args:

            method (str): passed directly to
                :py:func:`scipy.interpolate.griddata` and controls the method
                of interpolation used.
            **kwargs: Passed directly to :py:func:`scipy.interpolate.griddata`.

        This modifies the histogram, changing the data in-place. Bins are
        considered non-finite if the filled value or the uncertainty is ``nan``
        or ``inf``.

        """
        if not issubclass(self.data.dtype.type, Integral):
            finite = np.isfinite(self.data).ravel()
            if self.has_uncert:
                finite &= np.isfinite(self.uncert).ravel()
            if not all(finite):
                g = self.grid()
                points = np.vstack(g).reshape(self.dim, -1).T
                values = self.data.ravel()
                self.data[...] = interpolate.griddata(
                    points[finite],
                    values[finite],
                    points,
                    method=method,
                    **kwargs).reshape(self.shape)
                if self.has_uncert:
                    values = self.uncert.ravel()
                    self.uncert[...] = interpolate.griddata(
                        points[finite],
                        values[finite],
                        points,
                        method=method,
                        **kwargs).reshape(self.shape)

    def smooth(self, weight=0.5, sigma=1, mode='nearest', **kwargs):
        """Smooth the histogram using a Gaussian filter.

        Keyword Args:

            weight (float [0, 1]): Linear weighting for Gaussian filter. A
                value of 1 will replace the data with the actual result from
                :py:func:`scipy.ndimage.filters.gaussian_filter`.
            sigma (float or sequence of floats): Passed directly to
                :py:func:`scipy.ndimage.filters.gaussian_filter`.
            mode (str): Passed directly to
                :py:func:`scipy.ndimage.filters.gaussian_filter`.
            **kwargs: Passed directly to
                :py:func:`scipy.ndimage.filters.gaussian_filter`.

        All non-finite bins are filled using
        :py:meth:`Histogram.interpolate_nans` before the Gaussian filter is
        applied. If the underlying data is of integral type, it will be
        converted to `numpy.float64` before the filter is applied.
        """
        self.interpolate_nonfinites()
        if issubclass(self.data.dtype.type, Integral):
            self._data = self.data.astype(np.float64)
        Zf = ndimage.filters.gaussian_filter(self.data, sigma=sigma, mode=mode,
                                             **kwargs)
        self.data   = weight * Zf + (1. - weight) * self.data
        if self.has_uncert:
            Uf = ndimage.filters.gaussian_filter(self.uncert, sigma=sigma,
                                                 mode=mode, **kwargs)
            self.uncert = weight * Uf + (1. - weight) * self.uncert

### slicing and shape changing
    def slices_data(self, axis=0):
        """Iterable over the data along specified axis."""
        return np.rollaxis(self.data, axis)

    def slices_uncert(self, axis=0):
        """Iterable over the uncertainty along specified axis."""
        uncert = self.uncert
        return np.rollaxis(uncert, axis)

    def slices(self, axis=0):
        """Generator of histograms along specified axis."""
        if self.has_uncert:
            uncert_slices = self.slices_uncert(axis)
        else:
            uncert_slices = [None]*self.axes[axis].nbins
        for d, u in zip(self.slices_data(axis), uncert_slices):
            yield Histogram(
                *[a for i, a in enumerate(self.axes) if i != axis],
                data=d,
                uncert=u,
                title=self.title,
                label=self.label)

    def rebin(self, nbins=2, axis=0, snap='low', clip=True):
        """Create a new histogram with merged bins along an axis.

        Keyword Args:
            nbins (int): Number of bins to merge.
            axis (int): Axis along which to merge bins.
            snap (str): Controls edge behavior if `nbins` does not evenly
                divide the number of bins in this `axis`.
            clip (bool): Wether or not to include the non-uniform bin in the
                case that `bins` does not evenly divide the number of bins in
                this `axis`.
        """
        axnew = [ax.mergebins(nbins, snap, clip) if i == axis else ax
                 for i, ax in enumerate(self.axes)]

        if self.has_uncert:
            x = unp.uarray(self.data.astype(np.float64), copy(self.uncert))
        else:
            x = self.data.copy()

        x = np.rollaxis(x, axis, 0)
        a = x.shape[0]
        d, r = divmod(a, nbins)
        shp = [d, nbins]
        if len(x.shape) > 1:
            shp += list(x.shape[1:])
        if r == 0:
            x = x.reshape(shp)
        else:
            if not clip:
                shp[0] = shp[0] + r
                zeros = np.zeros([nbins - r] + list(x.shape[1:]))
                if snap == 'low':
                    x = np.concatenate((x, zeros))
                else:
                    x = np.concatenate((zeros, x))
            x = np.resize(x, shp)
        x = x.sum(1)
        x = np.rollaxis(x, 0, axis + 1)

        if self.has_uncert:
            data = unp.nominal_values(x)
            uncert = unp.std_devs(x)
        else:
            data = x
            uncert = None

        return Histogram(*axnew, title=self.title, label=self.label,
                         data=data, uncert=uncert)

    def cut(self, *args, **kwargs):
        """Truncate a histogram along one or more axes.

        To cut (truncate) a one dimensional histogram from 0 to 5,
        each of the following are valid. This will also cut an
        ND histogram along the first (x) axis::

            hcut = h.cut(0, 5)
            hcut = h.cut((0, 5))
            hcut = h.cut(0, 5, axis=0)
            hcut = h.cut((0, 5), axis=0)

        To cut along the second (y) axis, you can do the following::

            hcut = h.cut(0, 5, axis=1)
            hcut = h.cut((0, 5), axis=1)
            hcut = h.cut(None, (0, 5))
            hcut = h.cut((None, None), (0, 5))

        To cut only on one side, `None` may be used to indicate
        +/- infinity. This makes a lower-bound type cut at 0::

            hcut = h.cut(0, None)

        Finally, to cut on multiple dimensions at once, the cut
        ranges can be strung together. These examples cut
        along the first axis (x) from 0 to 5 and along the second
        axis (y) from 1 to 6::

            hcut = h.cut(0, 5, 1, 6)
            hcut = h.cut((0, 5), (1, 6))

        The first example above is useful for cutting 2D histogram
        using extent lists as used in other libraries like
        matplotlib::

            hcut = h.cut(*ax.extent())

        where, for example, `ax.extent()` returns the extent in
        the form::

            [xmin, xmax, ymin, ymax]
        """
        axis = kwargs.pop('axis', None)
        rng = []
        for a in args:
            if isinstance(a, Iterable):
                rng += a
                if len(a)== 1:
                    rng += [None]
            else:
                rng += [a]
        if (len(rng) % 2) == 1:
            rng.append(None)
        rng = np.asarray(rng)
        rng.shape = (-1, 2)

        if axis is not None:
            inrng = copy(rng)
            rng = [[None, None] for _ in range(self.dim)]
            rng[axis] = inrng[0]
            rng = np.asarray(rng)

        newaxes = []
        newdata = copy(self.data)
        if self.has_uncert:
            newuncert = copy(self.uncert)
        else:
            newuncert = None
        for i, (r, ax) in enumerate(zip(rng, self.axes)):
            xlow, xhigh = r
            if (xlow is None) and (xhigh is None):
                newaxes += [ax.copy()]
            else:
                a, m = ax.cut(xlow, xhigh, ('nearest', 'nearest'))
                indices = np.argwhere(m)[:, 0]
                newaxes += [a]
                newdata = newdata.take(indices, i)
                if newuncert is not None:
                    newuncert = newuncert.take(indices, i)

        return Histogram(*newaxes,
            data = newdata,
            uncert = newuncert,
            title = kwargs.get('title', copy(self.title)),
            label = kwargs.get('label', copy(self.label)))

    def occupancy(self, bins=100, limits=None, **kwargs):
        """Histogram the filled data of this histogram

        Returns a new histogram showing the occupancy of the data. This is
        effectively histograming the data points, ignoring the axes and
        uncertanties.
        """
        nans = np.isnan(self.data)
        if limits is None:
            limits = [self[~nans].min(), self[~nans].max()]
        ret = Histogram(bins, limits, **kwargs)
        ret.fill_from_sample(self[~nans].ravel())
        return ret

### curve fitting
    def fit(self, fcn, p0, **kwargs):
        """Fit a function to the histogram

        Fits the function ``fcn`` to the histogram, returning estimated
        parameters, their covariance matrix and a tuple containing the
        specified test result (chi-square test is default).
        """

        test = kwargs.pop('test', 'chisquare').lower()
        uncert = kwargs.pop('uncert', self.uncert)
        sel = kwargs.pop('sel', np.ones(self.data.shape, dtype=bool))

        if 'sigma' in kwargs:
            warn('"sigma" keyword not valid, use "uncert".')
        if 'abolute_sigma' in kwargs:
            warn('"absolue_sigma" keyword not used (always considered True).')

        # initial parameters
        if hasattr(p0, '__call__'):
            kwargs['p0'] = p0(self)
        else:
            kwargs['p0'] = copy(p0)

        npar = len(kwargs['p0'])

        ### Setup data selection
        sel &= np.isfinite(self.data)

        xx = self.grid()
        for x in xx:
            sel &= np.isfinite(x)
        xx = np.squeeze(tuple(x[sel].astype(np.float64) for x in xx))

        if uncert is not None:
            sel &= np.isfinite(uncert)

        if np.count_nonzero(sel) < npar:
            raise RuntimeError('Not enough data.')

        ## Setup data at grid points
        yy = self.data[sel].astype(np.float64)

        if uncert is not None:
            kwargs['sigma'] = uncert[sel].astype(np.float64)
            kwargs['absolute_sigma'] = True

        ### Do the fit
        pfit, pcov = opt.curve_fit(fcn, xx, yy, **kwargs)

        if not isinstance(pcov, np.ndarray):
            raise RuntimeError('Bad fit.')

        ### perform goodness of fit test
        if test not in [None, 'none']:
            N = len(xx)
            m = npar
            ndf = N - m
            yyfit = fcn(xx, *pfit)
            dyy = yy - yyfit

            # nchar is the minimum number of characters
            # used to disambiguate goodness of fit tests
            # at the moment, one letter is sufficient.
            nchar = 1
            if test[:nchar] == 'kstest'[:nchar]:
                # two-sided Kolmogorov-Smirov test
                D, pval = stats.kstest(dyy,
                            stats.norm(0, dyy.std()).cdf)
                ptest = (D, pval)
            elif test[:nchar] == 'shapiro'[:nchar]:
                # Shapiro-Wilk test
                W, pval = sp.stats.shapiro(dyy)
                ptest = (W, pval)
            else: # test[:nchar] == 'chisquare'[:nchar]:
                # simple Chi-squared test
                chisq, pval = stats.chisquare(yy, yyfit, len(pfit))
                ptest = (chisq/ndf, pval)

            return pfit, pcov, ptest

        return pfit, pcov
