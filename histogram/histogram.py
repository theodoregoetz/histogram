from __future__ import division, print_function

from copy import copy, deepcopy
import numpy as np
import itertools as it
from scipy import optimize as opt
from scipy import stats, ndimage, interpolate
from warnings import warn

from .histogram_axis import HistogramAxis
from .detail import isstr

# ignore divide by zero (silently create nan's)
np.seterr(divide='ignore', invalid='ignore')

class Histogram(object):
    '''N-dimensional histogram over a continuous range

    This is a histogram where each axis is
    a continuous (non-discrete) range with a set
    number of bins. The binning does not have to be
    evenly spaced.

    Args:
        axes (list): List of :py:class:`HistogramAxis` or constructor
            parameters thereof. These are the axis definitions.

    Keyword Args:
        label (str): Label for the filled data.
        title (str): Title of this histogram.
        data (scalar array): N-dimensional array for the filled data.
        uncert (scalar array): N-dimensional array for the uncertainty.
        dtype (scalar type): Type of the data array (data will be
        converted).

    Example:

        Typical usage would be to fill the histogram from a
        sample of data. In this example, we create a 1D
        histogram with 100 bins from 0 to 10, and fill it
        with 10k samples distributed normally around 5 with
        a width (sigma) of 1::

            import numpy as np
            from matplotlib import pyplot
            from histogram import Histogram

            h = Histogram(100, [0,10],'x (cm)','counts','Random Distribution')
            h.fill(np.random.normal(5,1,10000))
            fig,ax = pyplot.subplots(figsize=(4,2.5))
            fig.subplots_adjust(left=.18,bottom=.2,right=.95,top=.88)
            pt = ax.plothist(h,color='steelblue')
            pyplot.show()

        .. image:: images/histogram_1dnorm.png
    '''

    def __init__(self, *axes, **kwargs):

        if not axes:
            raise Exception('you must specify at least one axis.')

        self.axes = []
        labels = []
        i = 0
        while i < len(axes):
            if isinstance(axes[i], int):
                if len(axes) > (i+2) and isstr(axes[i+2]):
                    self.axes.append(HistogramAxis(*axes[i:i+3]))
                    i = i + 3
                else:
                    self.axes.append(HistogramAxis(*axes[i:i+2]))
                    i = i + 2
            elif isstr(axes[i]):
                labels.append(axes[i])
                i = i + 1
            elif isinstance(axes[i], HistogramAxis):
                self.axes.append(axes[i])
                i = i + 1
            else:
                self.axes.append(HistogramAxis(*axes[i]))
                i = i + 1

        self.label = kwargs.pop('label',None)
        self.title = kwargs.pop('title',None)
        if len(labels) > 0:
            if self.label is None:
                self.label = labels[0]
            else:
                raise Error('two labels given for this Histogram')
            if len(labels) > 1:
                if self.title is None:
                    self.title = labels[1]
                else:
                    raise Error('two titles given for this Histogram')

        shape = tuple(ax.nbins for ax in self.axes)

        data = kwargs.pop('data',None)
        dtype = kwargs.pop('dtype',None)

        if data is None:
            if dtype is None:
                dtype = np.int64
            self._data = np.zeros(shape=shape,dtype=dtype)
        else:
            data = np.asarray(data)
            assert data.shape == shape, 'Data shape must match axes.'
            if dtype is None:
                self._data = data
            else:
                self._data = data.astype(dtype)

        self.uncert = kwargs.pop('uncert',None)

### properties
    @property
    def data(self):
        ''':py:class:`numpy.ndarray` of the filled data

        The indexes are in the same order as the :py:class:`HistogramAxis` objects in the list stored in :py:attr:`Histogram.axes`. One can set this directly - shape is checked and data is written "in-place" when possible.

        Example:

            Here, we create a histogram and set the data directly::

                from scipy import stats
                from matplotlib import pyplot
                from histogram import Histogram

                h = Histogram(50, [0,10])

                xx, = h.grid
                h.data = 1000 * stats.norm(5,2).pdf(xx)

                fig,ax = pyplot.subplots(figsize=(4,2.5))
                fig.subplots_adjust(left=.15,bottom=.2,right=.9,top=.85)
                pt = ax.plothist(h,color='steelblue')
                pyplot.show()

        .. image:: images/histogram_data_1dnorm.png
        '''
        return self._data

    @data.setter
    def data(self,d):
        d = np.asarray(d)
        if self._data.dtype == d.dtype:
            self._data[...] = d
        else:
            assert d.shape == self.shape, 'Data shape mismatch.'
            self._data = d

    @property
    def uncert(self):
        ''':py:class:`numpy.ndarray` of the absolute uncertainty

        This has the same shape as :py:attr:`Histogram.data` or None. Under certain cases, this will be set automatically to the square-root of the data (Poisson statistics assumption).

        Example:

            When histogramming a sample of randomly distributed data, the uncertainty of each bin is the equal to the square-root of the counts in that bin::

                import numpy as np
                from scipy import stats
                from numpy import random as rand
                from matplotlib import pyplot
                from histogram import Histogram

                rand.seed(1)

                h = Histogram(30, [0,10])

                xx, = h.grid
                h.data = 1000 * stats.norm(5,2).pdf(xx)
                h.data += rand.normal(0,10,xx.shape)

                h.uncert = np.sqrt(h.data)

                fig,ax = pyplot.subplots(figsize=(4,2.5))
                fig.subplots_adjust(left=.15,bottom=.2,right=.9,top=.85)
                pt = ax.plothist(h,style='errorbar')
                pyplot.show()

        .. image:: images/histogram_uncert_1dnorm.png
        '''
        return getattr(self,'_uncert',None)

    @uncert.setter
    def uncert(self,u):
        if hasattr(self,'_uncert'):
            if u is None:
                del self._uncert
            else:
                self._uncert[...] = u
        elif u is not None:
            if hasattr(u,'__iter__'):
                u = np.asarray(u, dtype=np.float64)
                assert u.shape == self.shape, 'Uncertainty must have the same shape as the data.'
                self._uncert = u
            else:
                self._uncert = np.full(self.shape, u, dtype=np.float64)

    @property
    def uncert_ratio(self):
        '''The untertainty as ratios of the data'''
        if self.uncert is None:
            return None
        return self.uncert / self.data

    @uncert_ratio.setter
    def uncert_ratio(self, u):
        '''Set the uncertainty by ratio of the data'''
        self.uncert = (u * self.data).astype(np.float64)

    @property
    def title(self):
        '''Title of this histogram'''
        return getattr(self,'_title',None)

    @title.setter
    def title(self,t):
        if (t is None) or (t == ''):
            if hasattr(self,'_title'):
                del self._title
        elif not isstr(t):
            self._title = str(t)
        else:
            self._title = t

    @property
    def label(self):
        '''Label of the filled data in this histogram

        This is usually something like "counts".'''
        return getattr(self,'_label',None)

    @label.setter
    def label(self,l):
        if (l is None) or (l == ''):
            if hasattr(self,'_label'):
                del self._label
        elif not isstr(l):
            self._label = str(l)
        else:
            self._label = l

    def __eq__(self, that):
        '''Check if data and axes are equal

        Uncertainty, labels and titles are not considered. For complete equality, see :py:meth:`Histogram.isidentical`. For finer control of histogram comparison, consider using :py:func:`numpy.allclose()` on the data::

            import numpy as np
            from histogram import Histogram

            h1 = Histogram(10,(0,10))
            h2 = Histogram(10,(0,10))
            h3 = Histogram(10,(-10,10))

            h1.data[2] = 1
            h2.data[2] = 1
            h3.data[3] = 1

            # checks data and axes:
            assert h1 == h2
            assert not (h1 == h3)

            # checks only data:
            assert np.allclose(h1.data, h2.data)
        '''
        try:
            if not np.allclose(self.data,that.data):
                return False
            for a,aa in zip(self.axes,that.axes):
                if not (a == aa):
                    return False
        except ValueError:
            return False
        return True

    def isidentical(self, that):
        '''Check if histograms are identical including uncertainty and labels

        See also :py:meth:`Histogram.__eq__`.'''
        if not (self == that):
            return False
        if self.uncert is not None:
            if that.uncert is not None:
                if not np.allclose(self.uncert,that.uncert):
                    return False
            else:
                return False
        elif that.uncert is not None:
            return False
        if self.label != that.label:
            return False
        if self.title != that.title:
            return False
        for a,aa in zip(self.axes,that.axes):
            if a.label != aa.label:
                return False
        return True

### non-modifying information getters
    def __str__(self):
        '''Breif string representation of the data

        Returns the string representation of the numpy array containing the data only. Axes, uncertainty and labels are ignored.

        Example::

            from numpy import random as rand
            from histogram import Histogram

            rand.seed(1)

            h = Histogram(10,[0,10])
            h.fill(rand.normal(5,2,10000))
            print(h)

        output::

            [ 164  428  909 1484 1915 1934 1525  873  467  175]

        '''
        return str(self.data)

    def __call__(self,*xx,**kwargs):
        '''Value of histogram at a point

        Returns the value of the histogram at a specific point ``(x,y...)`` or array of points ``(xx,yy...)``.

        Args:
            xx (tuple of numbers or arrays): Point(s) inside the axes of this histogram.

        Keyword Args:
            overflow (number): Return value when the point lies outside this histogram. (default: 0)

        Example::

            from numpy import random as rand
            from histogram import Histogram

            rand.seed(1)

            h = Histogram(10,[0,10])
            h.fill(rand.normal(5,2,10000))
            for x in [0.5,1.5,2.5]:
                print( h(x) )

        output::

            164
            428
            909

        '''
        overflow = kwargs.pop('overflow',0)

        bin = []
        for x,ax in zip(xx,self.axes):
            b = ax.bin(x)
            if (b < 0) or (b >= ax.nbins):
                return overflow
            bin += [b]

        return self.data[tuple(bin)]

    def asdict(self):
        '''Dictionary representation of this histogram

        This includes uncertainty, axes, labels and title and is used to serialize the histogram to NumPy's binary format (see :py:func:`save_histogram_to_npz`).
        '''
        ret = {'data' : self.data}
        if self.uncert is not None:
            ret['uncert'] = self.uncert
        if self.label is not None:
            ret['label'] = self.label
        if self.title is not None:
            ret['title'] = self.title
        for i,a in enumerate(self.axes):
            e = 'edges{}'.format(i)
            el = 'label{}'.format(i)
            ret[e] = a.edges
            if a.label is not None:
                ret[el] = a.label
        return ret

    @staticmethod
    def fromdict(**kwargs):
        '''Create new :py:class:`Histogram` from a dictionary

        Required Keywords:

            * data
            * edges0, edges1 ... edgesN

        Optional Keywords:

            * uncert
            * label0, label1 ... labelN
            * label
            * title

        where ``data`` is an N-dimensional :py:class:`numpy.ndarray`, and edgei for i in [0,N] are 1-dimensional arrays specifying the edges along the ith dimension.

        ``uncert`` is either the same shape array as ``data`` or ``None``. ``label`` is the "counts" axis label of the histogram (string or ``None``), and ``title`` is the overall title of the histogram object (string or ``None``).

        Notes:

            This is not typically used to create new :py:class:`Histogram` objects, but is meant to be the counter-part to :py:meth:`Histogram.asdict` for deserialization.
        '''
        data = kwargs.pop('data')

        axes = []
        for i in range(len(data.shape)):
            e = 'edges{}'.format(i)
            el = 'label{}'.format(i)
            axes.append(HistogramAxis(
                kwargs.pop(e),
                label=kwargs.pop(el,None) ))

        return Histogram(*axes, data = data, **kwargs)

###    dimension and shape
    @property
    def dim(self):
        '''Dimension of this histogram (number of axes)'''
        return len(self.axes)

    @property
    def shape(self):
        '''Shape of the histogram data

        This is a tuple of the number of bins in each axis ``(x,y...)``. This is the same as :py:attr:`Histogram.data.shape`.
        '''
        return self.data.shape

    @property
    def size(self):
        '''Total number of bins (size of data)

        This is the product of the number of bins in each axis. This is the same as :py:meth:`Histogram.data.size`.
        '''
        return self.data.size()

###    axes information (bin edges and centers)
    def isuniform(self, rtol=1e-05, atol=1e-08):
        '''Check if all axes are uniform

        Returns "and" of :py:meth:`HistogramAxis.isuniform` for each axis.
        '''
        return all([ax.isuniform(rtol=rtol,atol=atol) for ax in self.axes])

    @property
    def edges(self):
        '''Edges of each axis as a tuple

        Output is in the form::

            ( [x0,x1..], [y0,y1...] ... )
        '''
        return tuple([ax.edges for ax in self.axes])

    @property
    def grid(self):
        '''Meshgrid of the bincenters of each axis

        This is a single array for 1D histograms - i.e. the bin-centers of the x-axis. For 2D histograms, this is a tuple of two 2D arrays::

            XX,YY = h2.grid

        Here, ``XX`` and ``YY`` are arrays of shape ``(xbins,ybins)``. For 1D histograms, the output is still a tuple so typically, one should expand this out with a comma::

            xx, = h1.grid
        '''
        if self.dim == 1:
            return (self.axes[0].bincenters,)
        else:
            centers = [ax.bincenters for ax in self.axes]
            return np.meshgrid(*centers, indexing='ij')

    @property
    def edge_grid(self):
        '''Meshgrid built from the axes' edges

        This is the same as :py:meth:`Histogram.grid` but for the edges of each axis instead of the bin centers.
        '''
        if self.dim == 1:
            return (self.axes[0].edges,)
        else:
            edges = [ax.edges for ax in self.axes]
            return np.meshgrid(*edges, indexing='ij')

    @property
    def binwidths(self):
        '''Widths of all bins along each axis

        This will always return a tuple::

            dx,dy = h2.binwidths

        Here, ``dx`` and ``dy`` are arrays of the widths of each bin along the `x` and `y` axes respecitively. For 1D histograms, the output is still a tuple so typically, one should expand this out with a comma::

            dx, = h1.binwidths
        '''
        return tuple([ax.binwidths for ax in self.axes])

    def binwidth(self, b=1, axis=0):
        '''Width of a specific bin ``b`` along an axis

        Args:
            b (int): Bin index (from zero, default: 1).
            axis (int): Axis index (default: 0).

        Note:

            Default is the second bin (index = 1) in the first (index = 0) axis.
        '''
        return self.axes[axis].binwidth(b)

    @property
    def binvolumes(self):
        '''Volumes of each bin

        Volume is defined as the product of the bin-widths along each axis for the given bin. For 1D histogram, this is the same as :py:attr:`Histogram.binwidths`. For 2D histograms, this returns a 2D array like the following where dxi is the width of the ith bin along the x-axis (first, index = 0)::

            [ [ dx0*dy0, dx0*dy1 ... ],
              [ dx1*dy0, dx1*dy1 ... ],
              ... ] = h2.binvolumes

            h.binvolumes[i,j] = dxi * dyj
        '''
        widths = self.binwidths
        if self.dim == 1:
            return widths
        else:
            return np.multiply.reduce(np.ix_(*widths))

    @property
    def overflow(self):
        '''Guaranteed overflow point when filling this histogram

        For 1D histograms, this is a tuple of one value ``(x,)`` generated by :py:attr:`HistogramAxis.overflow`. For 2D histograms, this will look like ``(x,y)``.

        Example::

            from histogram import Histogram

            ha = Histogram(10,[0,10])
            print(h)

            ha.fill(ha.overflow)
            print(h)

        Output::

            [0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0]
        '''
        return tuple(ax.overflow for ax in self.axes)

###    data information (limits, sum, extent)
    def sum(self, *axes):
        '''Sum of bin values or sum along one or more axes

        Optional Args:

            axes (tuple of integers): Axes to sum over.

        Returns the sum over all values or sums only over specific axes and returns a new :py:class:`Histogram` with reduced dimension.

        Example::

            from histogram import Histogram

            h = Histogram(10,[0,10])
            h.fill([1,2,2,3,3,3,4,4,4,4])

            print('sum of h:',h.sum())

            h2 = Histogram(10,[0,10],10,[0,10])
            h2.fill([1,2,2,3,3,3,4,4,4,4],
                    [1,1,1,1,1,1,2,2,2,2])

            print('h2 sum along axis 0:',h2.sum(0))

        Output::

            sum of h: 10
            h2 sum along axis 0: [0 6 4 0 0 0 0 0 0 0]
        '''
        if (len(axes)==0) or all(d in axes for d in range(self.dim)):
            return self.data.sum()

        ii = sorted(set(range(self.dim)) - set(axes))
        newaxes = [self.axes[i] for i in ii]

        newdata = np.apply_over_axes(np.sum, self.data, axes)
        newdata = np.squeeze(newdata)

        if self.uncert is None:
            newuncert = None
        else:
            uncratio_sq = np.power(self.uncert / self.data, 2)
            uncratio_sqsum = np.apply_over_axes(np.sum, uncratio_sq, axes)
            newuncert = np.sqrt(uncratio_sqsum) * newdata
            newuncert = np.squeeze(newuncert)

        return Histogram(
            *newaxes,
            data = newdata,
            uncert = newuncert,
            title = copy(self.title),
            label = copy(self.label))

    def integral(self,uncert=False):
        '''Total integral of the histogram

        This is the sum of each bin multiplied by the volume of the bins. If ``uncert=True`` is given, then the output is the 2-tuple: ``(integral, uncertainty)``.
        '''
        bv = self.binvolumes
        ret = np.sum(self.data * bv)
        if uncert:
            if self.uncert is None:
                e = np.sqrt(np.sum(self.data*bv*bv))
            else:
                e = np.sqrt(np.sum(self.uncert*self.uncert*bv*bv))
            ret = ret,e
        return ret

    def min(self,uncert=False):
        '''Minimum value of the filled data

        This will include the uncertainty if ``uncert=True``. Uncertainty is calculated as the square-root of the data if not already set.
        '''
        if not uncert:
            return np.nanmin(self.data)
        else:
            if self.uncert is None:
                self.uncert = np.sqrt(self.data)
            return np.nanmin(self.data - self.uncert)

    def max(self,uncert=False):
        '''Maximum value of the filled data

        This will include the uncertainty if ``uncert=True``. Uncertainty is calculated as the square-root of the data if not already set.
        '''
        if not uncert:
            return np.nanmax(self.data)
        else:
            if self.uncert is None:
                self.uncert = np.sqrt(self.data)
            return np.nanmax(self.data + self.uncert)

    def mean(self):
        '''Mean position of the data along the axes

        Returns:

            tuple: Mean (float) along each axis: ``(xmean,ymean...)``.

        This will ignore all ``nan`` and ``inf`` values in the histogram. Bin-centers are used as the position and non-equal widths are incorporated into the weighting of the results.
        '''
        mean = []
        for i,axis in enumerate(self.axes):
            data = self.data.sum(set(range(self.dim)) - {i})
            sel = np.isfinite(data)
            p = axis.bincenters[sel]
            w = data[sel]
            if not axis.isuniform():
                w *= axis.binwidths
            mean.append(np.average(p, weights=w))
        return tuple(mean)

    def var(self):
        '''Variance of the data along the axes

        Returns:

            tuple: Variance (float) along each axis: ``(xvar,yvar...)``.

        This will ignore all ``nan`` and ``inf`` values in the histogram. Bin-centers are used as the position and non-equal widths are incorporated into the weighting of the results.
        '''
        var = []
        for i,(axis,mean) in enumerate(zip(self.axes,self.mean())):
            data = self.data.sum(set(range(self.dim)) - {i})
            sel = np.isfinite(data)
            p = axis.bincenters[sel]
            w = data[sel]
            if not axis.isuniform():
                w *= axis.binwidths
            var.append(np.average((p-mean)**2, weights=w))
        return tuple(var)

    def std(self):
        '''Standard deviation of the data along the axes

        Returns:

            tuple: Standard deviation (float) along each axis: ``(xstd,ystd...)``.

        This is the square-root of the variance and will ignore all ``nan`` and ``inf`` values in the histogram. Bin-centers are used as the position and non-equal widths are incorporated into the weighting of the results.
        '''
        return tuple(np.sqrt(v) for v in self.var())

    def extent(self, maxdim=2, uncert=True, pad=None):
        '''Extent of axes and data

        Returns:

            tuple: extent like ``(xmin, xmax, ymin ...)``.

        By default, this includes the uncertainty if the last dimension is the histogram's data and not an axis::

            [xmin, xmax, ymin, ymax, ..., min(data-uncert), max(data+uncert)]

        padding is given as a percent of the actual extent of the axes or data (plus uncertainty) and can be either a single floating point number or a list of length ``2 * maxdim``.
        '''
        ext = []
        for ax in self.axes[:maxdim]:
            ext += [ax.min, ax.max]
        if len(ext) < (2*maxdim):
            ext += [self.min(uncert), self.max(uncert)]
        if pad is not None:
            if not hasattr(pad,'__iter__'):
                pad = [pad]*(2*maxdim)
            for dim in range(maxdim):
                a,b = 2*dim,2*dim+1
                w = ext[b] - ext[a]
                ext[a] -= pad[a] * w
                ext[b] += pad[b] * w
        return tuple(ext)

    def errorbars(self, maxdim=2, asratio=False):
        '''Bin half-widths and data uncertainties

        Keyword Args:

            maxdim (int): Number of dimensions to return (default: 2).
            asratio (bool): Return ratios instead of absolute values (default: False).

        Returns:

            list of arrays: Errors to be used for plotting this histogram.

        Example::

            from numpy import random as rand
            from matplotlib import pyplot
            from histogram import Histogram

            rand.seed(1)

            h = Histogram(10,[0,10])
            h.fill(rand.normal(5,1,500))

            x, = h.grid
            y = h.data
            xerr,yerr = h.errorbars()

            fig,ax = pyplot.subplots(figsize=(4,2.5))
            pt = ax.errorbar(x,y,xerr=xerr,yerr=yerr,ls='none')
            fig.tight_layout()

            pyplot.show()

        .. image:: images/histogram_errorbars.png
        '''
        if maxdim > self.dim:
            if self.uncert is None:
                self.uncert = np.sqrt(self.data)

        if asratio:
            ret = [0.5*ax.binwidths/(ax.max-ax.min)
                   for ax in self.axes[:maxdim]]
            if len(ret) < maxdim:
                ret += [self.uncert / (self.data.max() - self.data.min())]

        else:
            ret = [0.5*ax.binwidths for ax in self.axes[:maxdim]]
            if len(ret) < maxdim:
                ret += [self.uncert]

        return ret

    def asline(self, xlow=None, xhigh=None):
        '''Points describing this histogram as a line

        Arguments:
            xlow (float or None): Lower bound along axis.
            xhigh (float or None): Upper bound along axis.

        Returns:

            ..

            **tuple**: ``(x, y, extent)``

            * **x** (1D float array) - `x`-coordinate array.
            * **y** (1D float array) - `y`-coordinate array.
            * **extent** (tuple) - ``(xmin, xmax, ymin, ymax)``.

        Example::

            from numpy import random as rand
            from matplotlib import pyplot
            from histogram import Histogram

            rand.seed(1)

            h = Histogram(10,[0,10])
            h.fill(rand.normal(5,1,500))

            x,y,extent = h.asline()

            fig,ax = pyplot.subplots(figsize=(4,2.5))
            pt = ax.plot(x,y,lw=2)
            fig.tight_layout()

            pyplot.show()

        .. image:: images/histogram_asline.png

        '''
        assert self.dim == 1, 'only 1D histograms can be translated into a line.'

        x = self.axes[0].edges
        y = self.data

        xx = np.column_stack([x[:-1],x[1:]]).ravel()
        yy = np.column_stack([y,y]).ravel()

        if (xlow is not None) or (xhigh is not None):
            mask = np.ones(len(xx),dtype=np.bool)
            if xlow is not None:
                mask &= (xlow <= xx)
            if xhigh is not None:
                mask &= (xx < xhigh)
            if not mask.any():
                raise Exception('range is not valid')
            if not mask.all():
                xx = xx[mask]
                yy = yy[mask]

            a,b = None,None
            if not mask[0]:
                a = 1
            if not mask[-1]:
                b = -1
            if (a is not None) or (b is not None):
                xx = xx[a:b]
                yy = yy[a:b]

        extent = (min(xx), max(xx), min(yy), max(yy))
        return xx,yy,extent



    def aspolygon(self, xlow=None, xhigh=None, ymin=0):
        '''Return a polygon of the histogram

        Arguments:
            ymin (scalar, optional): Base-line for the polygon. Usually set to zero for histograms of integer fill-type.
            xlim (scalar 2-tuple, optional): Range in `x` to be used.

        Returns:

            ..

            **tuple**: ``(points, extent)``.

            * **points** (scalar array) - Points defining the polygon beginning and ending with the point ``(xmin, ymin)``.
            * **extent** (scalar tuple) - Extent of the resulting polygon in the form: ``[xmin, xmax, ymin, ymax]``.
        '''
        assert self.dim == 1, 'only 1D histograms can be translated into a polygon.'

        ymin = ymin if ymin is not None else extent[2]

        xx,yy,extent = self.asline(xlow,xhigh)

        xx = np.hstack([xx[0],xx,xx[-1],xx[0]])
        yy = np.hstack([ymin,yy,ymin,ymin])

        extent[2] = ymin
        return xx,yy,extent

### self modifying methods (set, fill)
    def __getitem__(self,*args):
        '''Direct access to the filled data'''
        return self.data.__getitem__(*args)

    def __setitem__(self,*args):
        '''Direct access to the filled data'''
        return self.data.__setitem__(*args)

    def set(self,val,uncert=None):
        '''Set filled data to specific value

        This will set the uncertainty to ``None`` by default and will accept a single value or an array the same shape as the data. Data will be cast to the data type already stored in the histogram.
        '''
        if isinstance(val, np.ndarray):
            self.data.T[...] = val.T
        else:
            self.data[...] = val

        if uncert is None:
            if self.uncert is not None:
                self.uncert = None
        else:
            if self.uncert is None:
                self.uncert = np.empty(self.data.shape)

            if isinstance(uncert, np.ndarray):
                self.uncert.T[...] = uncert.T
            else:
                self.uncert[...] = uncert

    def reset(self):
        '''Set data to zero and uncertainty to ``None``.'''
        self.set(0)
        self.uncert = None

    def fill(self,*args):
        '''Fill histogram with sample data

        Arguments (``\*args``) are the sample of data an optional associated weights. Weights may be a single number or an array of length `N`. The default (``None``) is equivalent to ``weights=1``. Example::

            from histogram import Histogram

            ### 1D Example
            h = Histogram(10,[0,10])

            h.fill(1)           # single value
            h.fill(2,2)         # single value with weight
            h.fill([3,3,3])     # data sample
            h.fill([4,4],2)     # data sample with constant weight
            h.fill([5,5],[2,3]) # data sample with variable weights

            print(h)
            # output:
            # [0 1 2 3 4 5 0 0 0 0]

            ### 2D Example
            h = Histogram(3,[0,3],10,[0,10])
            xdata = [0,0,1,1,2]
            ydata = [1,2,3,4,5]
            weights = [1,2,3,4,5]
            h.fill(xdata,ydata,weights)

            print(h)
            # output:
            # [[0 1 2 0 0 0 0 0 0 0]
            #  [0 0 0 3 4 0 0 0 0 0]
            #  [0 0 0 0 0 5 0 0 0 0]]
        '''
        if len(args) > self.dim:
            sample = args[:-1]
            weights = args[-1]
        else:
            sample = args
            weights = None

        self.fill_from_sample(sample,weights)

    def fill_one(self, pt, wt=1):
        '''Fill a single data point

        This increments a single bin by weight ``wt``. While it is the fastest method for a single entry, it should only be used as a last resort because its roughly 10 times slower than :py:meth:`Histogram.fill_from_sample` when filling many entries.
        '''
        try:
            if pt < self.axes[0].min or self.axes[0].max < pt:
                return
            self.data[self.axes[0].bin(pt)] += wt
        except ValueError:
            b = []
            for x,ax,m in zip(pt,self.axes,self.data.shape):
                if x < ax.min or ax.max < x:
                    return
                b += [ax.bin(x)]
            self.data[tuple(b)] += wt

    def fill_from_sample(self, sample, weights=None):
        '''Fill histogram from sample of data

        This fills the histogram from sample with shape `(D,N)` array where `D` is the dimension of the histogram and `N` is the number of points to fill. The optional ``weights`` may be a single number or an array of length `N`. The default (``None``) is equivalent to ``weights = 1``.

        This is the primary work-horse of the :py:class:`Histogram` class and should be favored, along with the wrapper method :py:meth:`Histogram.fill` over :py:meth:`Histogram.fill_one`.

        Example::

            from numpy import random as rand
            from matplotlib import pyplot
            from histogram import Histogram

            rand.seed(1)

            ### 1D Example
            h = Histogram(10,[0,10], 30,[0,10])

            h.fill_from_sample(rand.normal(5,1,(2,20000)))

            fig,ax = pyplot.subplots(figsize=(4,2.5))
            pt = ax.plothist(h)
            fig.tight_layout()

            pyplot.show()

        .. image:: images/histogram_fill_from_sample.png
        '''
        if not isinstance(sample, np.ndarray):
            sample = np.array(sample)
        wt = weights
        if weights is not None:
            if not hasattr(weights,'__iter__'):
                wt = np.empty((sample.T.shape[0],))
                wt[...] = weights
        h,e = np.histogramdd(sample.T, self.edges, weights=wt)
        self.data += h.astype(self.data.dtype)

### operations
    def clone(self,dtype=None,**kwargs):
        '''Create a complete copy of this histogram'''
        if dtype is None:
            new_data = self.data.copy()
        else:
            new_data = self.data.astype(dtype)

        if self.uncert is None:
            new_uncert = None
        else:
            new_uncert = self.uncert.copy()

        return Histogram(
            *[ax.clone() for ax in self.axes],
            data = new_data,
            uncert = new_uncert,
            title = kwargs.get('title',copy(self.title)),
            label = kwargs.get('label',copy(self.label)))

    def added_uncert(self, that):
        '''Added uncertainty by absolute values

        Returns the uncertainty added directly in quadrature with another histogram. This assumes Poisson statistics (``uncert = sqrt(data)``) if ``uncert is None``.
        '''
        if not isinstance(that,Histogram):
            return self.uncert

        if self.uncert is None:
            self_unc_sq = self.data.astype(np.float64)
        else:
            self_unc_sq = np.power(self.uncert,2)

        if that.uncert is None:
            that_unc_sq = that.data.astype(np.float64)
        else:
            that_unc_sq = np.power(that.uncert,2)

        self_unc_sq.T[...] += that_unc_sq.T

        return np.sqrt(self_unc_sq)

    def added_uncert_ratio(self, that, nans=0):
        '''Added uncertainty by ratios of the data

        Returns the uncertainty added by ratio and in quadrature with another histogram. This assumes Poisson statistics (``uncert = sqrt(data)``) if ``uncert is None``.
        '''
        if not isinstance(that,Histogram):
            return self.uncert

        data = np.asarray(self.data, dtype=np.float64)

        if self.uncert is None:
            self_unc_ratio_sq = 1. / np.abs(data)
        else:
            self_unc_ratio_sq = np.power(self.uncert / data, 2)

        data = np.asarray(that.data, dtype=np.float64)

        if that.uncert is None:
            that_unc_ratio_sq = 1. / np.abs(data)
        else:
            that_unc_ratio_sq = np.power(that.uncert.T / data.T, 2).T

        self_unc_ratio_sq.T[...] += that_unc_ratio_sq.T

        return np.sqrt(self_unc_ratio_sq)

    def clear_nans(self, val=0):
        '''Set all non-finite bins to a specific value'''
        isnan = np.isnan(self.data) | np.isinf(self.data)
        self.data[isnan] = val
        if self.uncert is not None:
            isnan = np.isnan(self.uncert) | np.isinf(self.uncert)
            self.uncert[isnan] = val

    def dtype(self,that,div=False):
        '''Return dtype that should result from a binary operation

        Default is the result for addition, subtraction or multiplication. With ``div = True``, this will always return ``float64``.'''
        if div:
            return np.float64

        if isinstance(that, Histogram):
            that_dtype = that.data.dtype
        else:
            that_dtype = np.dtype(type(that))

        if self.data.dtype == that_dtype:
            return None
        else:
            inttypes = [np.int32, np.int64]
            if (self.data.dtype in inttypes) and \
               (that_dtype in inttypes):
                return np.int64
            else:
                return np.float64

    def __iadd__(self,that):
        '''In-place addition

        Uncertainty is not modified if both uncertainties are ``None``.'''
        if isinstance(that,Histogram):
            if any(u is not None for u in [self.uncert,that.uncert]):
                self.uncert = self.added_uncert(that)
            self.data[...] += that.data
        elif hasattr(that,'__iter__'):
            self.data.T[...] += np.asarray(that).T
        else:
            self.data += that
        return self

    def __radd__(self, that):
        '''Commuting addition'''
        return self + that

    def __add__(self, that):
        '''Addition'''
        ret = self.clone(self.dtype(that))
        ret += that
        return ret

    def __isub__(self, that):
        '''In-place subtraction

        If the uncertainties are ``None``, then Poisson statistics are assumed.'''
        unc = self.added_uncert(that)

        if isinstance(that,Histogram):
            self.data.T[...] -= that.data.T
        elif isinstance(that,np.ndarray):
            self.data.T[...] -= that.T
        else:
            self.data -= that

        self.uncert = unc
        self.clear_nans()
        return self

    def __rsub__(self, that):
        '''Commuting subtraction'''
        ret = self.clone(self.dtype(that))
        ret.data = that - ret.data
        return ret

    def __sub__(self, that):
        '''Subtraction'''
        ret = self.clone(self.dtype(that))
        ret -= that
        return ret

    def __imul__(self, that):
        '''In-place multiplication'''
        if isinstance(that,Histogram):
            uncrat = self.added_uncert_ratio(that)
            self.data.T[...] *= np.asarray(that.data, dtype=np.float64).T
        else:
            if self.uncert is None:
                uncrat = 1. / np.sqrt(np.abs(self.data))
            else:
                uncrat = self.uncert / np.abs(self.data)

            if hasattr(that,'__iter__'):
                self.data.T[...] *= np.asarray(that, dtype=np.float64).T
            else:
                self.data *= that

        self.uncert = uncrat * self.data
        self.clear_nans()

        return self

    def __rmul__(self, that):
        '''Commuting mulitplication'''
        return self * that

    def __mul__(self, that):
        '''Multiplication'''
        ret = self.clone(self.dtype(that))
        ret *= that
        return ret

    def __itruediv__(self, that):
        '''In-place (true) division

        If the uncertainties are ``None``, then Poisson statistics are assumed. After division, ``nan`` and ``inf`` values are set to zero.'''
        if isinstance(that,Histogram):
            uncrat = self.added_uncert_ratio(that)
            self.data.T[...] /= np.asarray(that.data, dtype=np.float64).T
        else:
            if self.uncert is None:
                uncrat = 1. / np.sqrt(np.abs(self.data))
            else:
                uncrat = self.uncert / np.abs(self.data)

            if hasattr(that,'__iter__'):
                self.data.T[...] /= np.asarray(that, dtype=np.float64).T
            else:
                self.data /= that

        self.uncert = uncrat * self.data
        self.clear_nans()

        return self

    def __rtruediv__(self, that):
        '''Commuting (true) division

            that = 1.
            hret = that / hself

        If the uncertainties are ``None``, then Poisson statistics are assumed. After division, ``nan`` and ``inf`` values are set to zero.
        '''
        uncrat = self.added_uncert_ratio(that)
        ret = self.clone(self.dtype(that,div=True))
        ret.data = that / ret.data
        ret.uncert = uncrat * ret.data
        self.clear_nans()
        return ret

    def __truediv__(self, that):
        '''Division'''
        ret = self.clone(self.dtype(that,div=True))
        ret /= that
        return ret

### interpolating and smoothing
    def interpolate_nans(self, method='cubic', **kwargs):
        '''Replace non-finite (nan or inf) bins with interpolated values

        Keyword Args:

            method (str): passed directly to :py:func:`scipy.interpolate.griddata` and controls the method of interpolation used.
            **kwargs: Passed directly to :py:func:`scipy.interpolate.griddata`.

        This modifies the histogram, changing the data in-place. Bins are considered non-finite if the filled value or the uncertainty is ``nan`` or ``inf``.
        '''
        finite = np.isfinite(self.data).ravel()
        if self.uncert is not None:
            finite &= np.isfinite(self.uncert).ravel()

        if not all(finite):
            g = self.grid
            points = np.vstack(g).reshape(self.dim,-1).T

            values = self.data.ravel()

            self.data[...] = interpolate.griddata(
                points[finite],
                values[finite],
                points,
                **kwargs).reshape(self.shape)

            if self.uncert is not None:
                values = self.uncert.ravel()
                self.uncert[...] = interpolate.griddata(
                    points[finite],
                    values[finite],
                    points,
                    **kwargs).reshape(self.shape)

        return self

    def smooth(self, weight=0.5, mode='nearest'):
        '''Create a new smoothed histogram using a Gaussian filter

        Keyword Args:

            weight (float [0,1]): Linear weighting for Gaussian filter. A value of 1 will replace the data with the actual result from :py:func:`scipy.ndimage.filters.gaussian_filter`.
            mode (str): Passed directly to :py:func:`scipy.ndimage.filters.gaussian_filter`.

        All non-finite bins are filled using :py:meth:`Histogram.interpolate_nans` before the Gaussian filter is applied.
        '''
        hnew = self.clone().interpolate_nans()

        Zf = ndimage.filters.gaussian_filter(hnew.data,1,mode=mode)
        hnew.data   = weight*Zf + (1.-weight)*hnew.data

        if hnew.uncert is not None:
            Uf = ndimage.filters.gaussian_filter(hnew.uncert,1,mode=mode)
            hnew.uncert = weight*Uf + (1.-weight)*hnew.uncert

        return hnew

### slicing and shape changing
    def slices(self, axis=0):
        axes = []
        for i in range(self.dim):
            if i != axis:
                axes.append(self.axes[i])

        data_slices = self.slices_data(axis)
        uncert_slices = self.slices_uncert(axis)

        for d,u in zip(data_slices,uncert_slices):
            yield Histogram(
                *axes,
                data = d,
                uncert = u,
                title = copy(self.title),
                label = copy(self.label))

    def slices_data(self, axis=0):
        if axis == 0:
            return self.data
        else:
            return np.rollaxis(self.data,axis)

    def slices_uncert(self, axis=0):
        if self.uncert is None:
            self.uncert = np.sqrt(self.data)
        if axis == 0:
            return self.uncert
        else:
            return np.rollaxis(self.uncert,axis)

    def rebin(self, nbins=2, axis=0, snap='low', clip=True):
        '''
        nbins describes the number of bins to merge
        along each axis, or along one axis specified by
        the argument "axis".
        '''
        if not hasattr(nbins,'__iter__'):
            ones = [1]*self.dim
            ones[axis] = nbins
            nbins = ones
        if not hasattr(clip,'__iter__'):
            clip = [clip for _ in range(self.dim)]

        def _rebin(x,nbins,clip):
            xx = x.copy()
            for i,(n,c) in enumerate(zip(nbins,clip)):
                if n > 1:
                    xx = np.rollaxis(xx,i,0)
                    a = xx.shape[0]
                    d,r = divmod(a,n)
                    shp = [d,n]
                    if len(xx.shape) > 1:
                        shp += list(xx.shape[1:])
                    if r == 0:
                        xx = xx.reshape(shp)
                    else:
                        if not c:
                            shp[0] = shp[0] + r
                        xx.resize(shp)
                    xx = xx.sum(1)
                    xx = np.rollaxis(xx,0,i+1)
            return xx

        axnew = [ax.mergebins(n,snap,clip) for ax,n in zip(self.axes,nbins)]

        hnew = Histogram(*axnew,
            data = _rebin(self.data, nbins, clip),
            title=self.title,
            label=self.label)

        if self.uncert is not None:
            unc_ratio_sq = (self.uncert / self.data)**2
            unc_ratio_sq = _rebin(unc_ratio_sq, nbins, clip)
            hnew.uncert = np.sqrt(unc_ratio_sq) * hnew.data

        return hnew

    def cut(self, *args, **kwargs):
        '''Truncate a histogram along one or more axes.

        To cut (truncate) a one dimensional histogram from 0 to 5,
        each of the following are valid. This will also cut an
        ND histogram along the first (x) axis::

            hcut = h.cut(0,5)
            hcut = h.cut((0,5))
            hcut = h.cut(0,5, axis=0)
            hcut = h.cut((0,5), axis=0)

        To cut along the second (y) axis, you can do the following::

            hcut = h.cut(0,5, axis=1)
            hcut = h.cut((0,5), axis=1)
            hcut = h.cut(None, (0,5))
            hcut = h.cut((None,None), (0,5))

        To cut only on one side, `None` may be used to indicate
        +/- infinity. This makes a lower-bound type cut at 0::

            hcut = h.cut(0,None)

        Finally, to cut on multiple dimensions at once, the cut
        ranges can be strung together. These examples cut
        along the first axis (x) from 0 to 5 and along the second
        axis (y) from 1 to 6::

            hcut = h.cut(0,5,1,6)
            hcut = h.cut((0,5),(1,6))

        The first example above is useful for cutting 2D histogram
        using extent lists as used in other libraries like
        matplotlib::

            hcut = h.cut(*ax.extent())

        where, for example, `ax.extent()` returns the extent in
        the form::

            [xmin,xmax,ymin,ymax]
        '''
        axis = kwargs.pop('axis',None)
        rng = []
        for a in args:
            if hasattr(a,'__iter__'):
                if (len(rng) % 2) == 1:
                    rng += [None]
                rng += a
            else:
                rng += [a]
        if (len(rng) % 2) == 1:
            rng.append(None)
        rng = np.asarray(rng)
        rng.shape = (-1,2)

        if axis is not None:
            inrng = copy(rng)
            rng = [[None,None] for _ in range(self.dim)]
            rng[axis] = inrng[0]
            rng = np.asarray(rng)

        newaxes = []
        newdata = copy(self.data)
        newuncert = copy(self.uncert)
        for i,(r,ax) in enumerate(zip(rng,self.axes)):
            xlow,xhigh = r
            if (xlow is None) and (xhigh is None):
                newaxes += [ax.clone()]
            else:
                a,m = ax.cut(xlow,xhigh,('nearest','nearest'))
                indices = np.argwhere(m)[:,0]
                newaxes += [a]
                newdata = newdata.take(indices,i)
                if newuncert is not None:
                    newuncert = newuncert.take(indices,i)

        ''' DEBUG
            print('\n')
            print('args:',args)
            print('oldedges:')
            for a in self.axes:
                print('    ',len(a.edges),a.edges)
            print('newedges:')
            for a in newaxes:
                print('    ',len(a.edges),a.edges)
            print('olddata shape:',self.data.shape)
            print('newdata shape:',newdata.shape)
        '''

        return Histogram(*newaxes,
            data = newdata,
            uncert = newuncert,
            title = kwargs.get('title',copy(self.title)),
            label = kwargs.get('label',copy(self.label)))

    def cut_data(self, *rng, **kwargs):
        raise Exception('not implemented')

    def projection(self, axis=0):
        '''
        projection onto a single axis
        '''
        sumaxes = list(range(self.dim))
        sumaxes.remove(axis)

        newdata = np.apply_over_axes(np.sum, self.data, sumaxes).ravel()

        if self.uncert is None:
            newuncert = None
        else:
            uncratio_sq = np.power(self.uncert / self.data, 2)
            uncratio_sqsum = np.apply_over_axes(np.sum, uncratio_sq, sumaxes).ravel()
            newuncert = np.sqrt(uncratio_sqsum) * newdata

        return Histogram(
            self.axes[axis].clone(),
            data = newdata,
            uncert = newuncert,
            title = copy(self.title),
            label = copy(self.label))

    def occupancy(self,bins=100,rng=None,**kwargs):
        '''
        returns a new histogram showing the occupancy of the
        data. This is effectively histograming the data points.
        '''
        nans = np.isnan(self.data)
        if rng is None:
            rng = [self[~nans].min(),self[~nans].max()]
        ret = Histogram(bins,rng,**kwargs)
        ret.fill_from_sample(self[~nans].ravel())
        return ret

### curve fitting
    def fit(self, fcn, p0, **kwargs):
        '''
        Fits the function fcn to the histogram, returning
        estimated parameters, their covariance matrix and
        a tuple containing the specified test result
        (chi-square test is default).
        '''

        test = kwargs.pop('test','chisquare').lower()
        uncert = kwargs.pop('uncert',self.uncert)
        sel = kwargs.pop('sel',np.ones(self.data.shape,dtype=bool))

        if 'sigma' in kwargs:
            warn('"sigma" keyword not accepted, use "uncert".')
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

        xx = self.grid
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
        pfit,pcov = opt.curve_fit(fcn, xx, yy, **kwargs)

        if not isinstance(pcov, np.ndarray):
            raise RuntimeError('Bad fit.')

        ### perform goodness of fit test
        if test not in [None,'none']:
            N = len(xx)
            m = npar
            ndf = N - m
            yyfit = fcn(xx,*pfit)
            dyy = yy - yyfit

            # nchar is the minimum number of characters
            # used to disambiguate goodness of fit tests
            # at the moment, one letter is sufficient.
            nchar = 1
            if test[:nchar] == 'kstest'[:nchar]:
                # two-sided Kolmogorov-Smirov test
                D,pval = stats.kstest(dyy,
                            stats.norm(0,dyy.std()).cdf)
                ptest = (D,pval)
            elif test[:nchar] == 'shapiro'[:nchar]:
                # Shapiro-Wilk test
                W,pval = sp.stats.shapiro(dyy)
                ptest = (W,pval)
            else: # test[:nchar] == 'chisquare'[:nchar]:
                # simple Chi-squared test
                chisq,pval = stats.chisquare(yy,yyfit,len(pfit))
                ptest = (chisq/ndf,pval)

            return pfit,pcov,ptest

        return pfit,pcov
