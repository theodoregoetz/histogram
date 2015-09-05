from __future__ import division, print_function

from copy import copy, deepcopy
import numpy as np
import itertools as it
from scipy import optimize as opt
from scipy import stats, ndimage, interpolate

from .histogram_axis import HistogramAxis
from .detail import isstr

# ignore divide by zero (silently create nan's)
np.seterr(divide='ignore', invalid='ignore')

class Histogram(object):
    '''An N-dimensional histogram over a continuous range.

    This is an ND histogram where each axis is
    a continuous (non-discrete) range with a set
    number of bins.

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
        ''':py:class:`numpy.ndarray` of the filled data for this :py:class:`Histogram`.

        The indexes are in the same order as the :py:class:`HistogramAxis` objects in the list stored in :py:attr:`Histogram.axes`. One can set this directly - shape is checked and data is written "in-place" when possible.

        Example:

            Here, we create a histogram and set the data directly::

                from scipy import stats
                from matplotlib import pyplot
                from histogram import Histogram

                h = Histogram(50, [0,10])

                xx = h.grid
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
    def title(self):
        '''Title (string) of this histogram'''
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
        '''The (string) label of the filled data in this histogram. Usually something like "counts".'''
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
        if not (self == that):
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
        '''
        the str() representation of the numpy array containing
        the data only (axes, uncertainty and labels are ignored).
        '''
        return str(self.data)

    def __call__(self,*xx,**kwargs):
        '''
        the value of the histogram at a specific
        point (x,y..) or array of points (xx,yy...)

        optional arguements:
            overflow - return value when the point lies outside this histogram. (default: 0)
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
        '''
        convert the dict object to a Histogram using these keys:

        required keywords:

            * data
            * edges0, edges1 ... edgesN

        optional keywords:

            * label0, label1 ... labelN
            * uncert
            * label
            * title

        where `data` is an N-dimensional numpy.ndarray, and edgei
        for i in [0,N] are 1-dimensional arrays specifying the edges
        along the ith dimension.

        `uncert` is either the same shape array as `data` or None.
        `label` is the "counts" axis label of the histogram (string
        or None), and `title` is the overall title of the histogram
        object (string or None).
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
        '''
        the dimension of this histogram (number of axes)
        '''
        return len(self.data.shape)

    @property
    def shape(self):
        '''
        the shape of this histogram. This is a tuple
        of the number of bins in each axis (x,y...)
        '''
        return self.data.shape

    @property
    def size(self):
        '''
        the total number of bins. This is the product
        of the number of bins in each axis
        '''
        return self.data.size()

###    axes information (bin edges and centers)
    def isuniform(self,*args,**kwargs):
        '''
        true or false depending on if all
        axes are uniform (to within a certain tolerace)
        see HistogramAxis.isuniform() for more details
        '''
        return all([ax.isuniform(*args,**kwargs) for ax in self.axes])

    @property
    def edges(self):
        '''
        the edges of each axis a tuple:
            ( [x0,x1..], [y0,y1...] ... )
        '''
        return tuple([ax.edges for ax in self.axes])

    @property
    def grid(self):
        '''
        a tuple of the "meshgrid" of the bincenters of each axis.
        This is just a single array for 1D histograms - i.e. the
        bin-centers of the x-axis. For 2D histograms, this is a
        tuple of two 2D arrays:

            XX,YY = h2.grid()

        Here, XX and YY are arrays of shape (xbins,ybins...)
        '''
        if self.dim == 1:
            return self.axes[0].bincenters
        elif self.dim == 2:
            centers = [ax.bincenters for ax in self.axes]
            return np.meshgrid(*centers, indexing='ij')
        else:
            raise Exception('not implemented')

    @property
    def edge_grid(self):
        '''Return the meshgrid built from the axes' edges.

        This is just a single array for 1D histograms - i.e. the
        edges of the x-axis. For 2D histograms, this is a
        tuple of two 2D arrays:

            XX,YY = h2.grid()

        Here, XX and YY are arrays of shape (xedges,yedges...)
        '''
        if self.dim == 1:
            return self.axes[0].edges
        elif self.dim == 2:
            edges = [ax.edges for ax in self.axes]
            return np.meshgrid(*edges, indexing='ij')
        else:
            raise Exception('not implemented')

    @property
    def binwidths(self):
        '''
        tuple (or single array for 1D histogram) of the binwidths
        of each axis
        '''
        if self.dim == 1:
            return self.axes[0].binwidths
        else:
            return tuple([ax.binwidths for ax in self.axes])

    def binwidth(self, b=1, axis=0):
        '''
        the bin-width for a specific bin b on the axis
        specified. default is the first (0) x-axis
        '''
        return self.axes[axis].binwidth(b)

    @property
    def binvolumes(self):
        '''
        array of the volume of each bin. Volume is defined as the
        product of the bin-widths along each axis for the given
        bin.

        For 1D histogram, this is the same as h.binwidths

        for 2D histograms, this returns a 2D array like the
        following where dxi is the width of the ith bin along
        the x-axis (first, index = 0)

            [ [ dx0*dy0, dx0*dy1 ... ],
              [ dx1*dy0, dx1*dy1 ... ],
              ... ]

            h.binvolumes[i,j] = dxi * dyj
        '''
        widths = self.binwidths
        if self.dim == 1:
            return widths
        else:
            return np.multiply.reduce(np.ix_(*widths))

    @property
    def overflow(self):
        '''
        a value that is guaranteed to be an overflow
        when filling this histogram.
        '''
        return tuple(ax.overflow for ax in self.axes)

###    data information (limits, sum, extent)
    def sum(self, *axes, **kwargs):
        '''
        summation over all values or specific axes, reducing
        the dimensionality of the histogram
        '''
        if (len(axes) > 0) and (len(axes) < self.dim):
            ret = self.sum_over_axes(*axes)
        else:
            uncert = kwargs.get('uncert',False)
            ret = self.data.sum()
            if uncert:
                if self.uncert is None:
                    e = np.sqrt(ret)
                else:
                    e = np.sqrt(np.sum(self.uncert*self.uncert))
                ret = ret,e
        return ret

    def integral(self,uncert=False):
        '''
        total integral of the histogram. This is the sum
        multiplied by the bin volumes
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
        if not uncert:
            return np.nanmin(self.data)
        else:
            if self.uncert is None:
                self.uncert = np.sqrt(self.data)
            return np.nanmin(self.data - self.uncert)

    def max(self,uncert=False):
        if not uncert:
            return np.nanmax(self.data)
        else:
            if self.uncert is None:
                self.uncert = np.sqrt(self.data)
            return np.nanmax(self.data + self.uncert)

    def mean(self,axis=None):
        '''
        mean of the all values in this histogram
        '''
        return self.data[~np.isnan(self.data)].mean(axis)

    def std(self):
        '''
        standard deviation of all values in this histogram
        '''
        return self.data[np.isfinite(self.data)].std()

    def extent(self, maxdim=2, uncert=True, pad=None):
        '''
        extent of axes (up to maxdim) returned as
        a single tuple. By default, this includes
        the uncertainty if the last dimension is
        the histogram's data and not an axis:

            [xmin, xmax, ymin, ymax, ..., zmin-dz, zmax+dz]

        padding is given as a percent of the actual
        extent of the axes or data (plus uncertainty)
        and can be either a single floating point
        number or a list of length (2 * maxdim)
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
        if maxdim < self.dim:
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
        '''Return points describing this histogram as a line

        Arguments:
            xlim (scalar 2-tuple, optional): Range in `x` to be used.

        Returns:

            ..

            **tuple**: (`x`, `y`)

            x
                `(scalar array)` x-position of points
            y
                `(scalar array)` y-position of points
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

        extent = [min(xx), max(xx), min(yy), max(yy)]
        return xx,yy,extent



    def aspolygon(self, xlow=None, xhigh=None, ymin=0):
        '''Return a polygon of the histogram

        Arguments:
            ymin (scalar, optional): Base-line for the polygon. Usually set to zero for histograms of integer fill-type.
            xlim (scalar 2-tuple, optional): Range in `x` to be used.

        Returns:

            ..

            **tuple**: (``points``, ``extent``).

            points
                `(scalar array)` Points defining the polygon beginning and ending with the point ``(xmin,ymin)``.

            extent
                `(scalar tuple)` Extent of the resulting polygon in the form: ``[xmin, xmax, ymin, ymax]``.
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
        return self.data.__getitem__(*args)

    def __setitem__(self,*args):
        return self.data.__setitem__(*args)

    def set(self,val,uncert=None):
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
        self.set(0)

    def fill(self,*args):
        '''
        convert \*args to sample, weights
        where sample is of shape (D,N)
        where D is the dimension of the
        histogram and N is the number of
        points to fill. Note that this
        will be transposed just before
        sending to np.histogramdd

        weights may be a single number or
        an array of length N. The default
        (None) is equivalent to weights=1
        '''
        if len(args) > self.dim:
            sample = args[:-1]
            weights = args[-1]
        else:
            sample = args
            weights = None

        self.fill_from_sample(sample,weights)

    def fill_one(self, pt, wt=1):
        '''
        increment a single bin by weight wt.
        This should be used as a last resort
        because it is roughly 10 times
        slower than fill_from_sample() when
        filling many entries.
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
        '''
        fills the histogram from sample
        which must be (D,N) array where
        D is the dimension of the histogram
        and N is the number of points
        to fill.

        weights may be a single number or
        an array of length N. The default
        (None) is equivalent to weights=1
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
        '''
        add the uncertainties directly in quadrature
        assume uncert = sqrt(data) if uncert is None
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
        '''
        add the uncertainties by ratio to the data in quadrature
        assume uncert = sqrt(data) if uncert is None
        '''

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
        isnan = np.isnan(self.data) | np.isinf(self.data)
        self.data[isnan] = val
        if self.uncert is not None:
            isnan = np.isnan(self.uncert) | np.isinf(self.uncert)
            self.uncert[isnan] = val

    def dtype(self,that,div=False):
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
        if isinstance(that,Histogram):
            self.uncert = self.added_uncert(that)
            self.data[...] += that.data
        elif hasattr(that,'__iter__'):
            self.data.T[...] += np.asarray(that).T
        else:
            self.data += that
        return self

    def __radd__(self, that):
        return self + that

    def __add__(self, that):
        ret = self.clone(self.dtype(that), title=r'', label=r'')
        ret += that
        return ret

    def __isub__(self, that):
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
        '''
            that = 1.
            hret = that - hself
        '''
        ret = self.clone(self.dtype(that), title=r'', label=r'')
        ret.data = that - ret.data
        return ret

    def __sub__(self, that):
        ret = self.clone(self.dtype(that), title=r'', label=r'')
        ret -= that
        return ret

    def __imul__(self, that):

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
        return self * that

    def __mul__(self, that):
        ret = self.clone(self.dtype(that), title=r'', label=r'')
        ret *= that
        return ret

    def __itruediv__(self, that):

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
        '''
            that = 1.
            hret = that / hself
        '''
        uncrat = self.added_uncert_ratio(that)
        ret = self.clone(self.dtype(that,div=True), title=r'', label=r'')
        ret.data = that / ret.data
        ret.uncert = uncrat * ret.data
        self.clear_nans()
        return ret

    def __truediv__(self, that):
        ret = self.clone(self.dtype(that,div=True), title=r'', label=r'')
        ret /= that
        return ret

### interpolating and smoothing
    def interpolate_nans(self, **kwargs):
        if not 'method' in kwargs:
            kwargs['method'] = 'cubic'

        hnew = self.clone()
        s = hnew.data.shape

        hasnan = np.isnan(hnew.data).any()
        if hnew.uncert is not None:
            hasnan |= np.isnan(hnew.uncert).any()

        if hasnan:
            g = self.grid
            points = np.vstack(g).reshape(self.dim,-1).T

            values = self.data.ravel()
            nans = np.isnan(values)

            hnew.data = interpolate.griddata(
                points[~nans],
                values[~nans],
                points,
                **kwargs).reshape(s)

            if self.uncert is not None:
                values = self.uncert.ravel()
                nans = ((values/self.data.ravel())<0.001) | np.isnan(values)
                hnew.uncert = interpolate.griddata(
                    points[~nans],
                    values[~nans],
                    points,
                    **kwargs).reshape(s)

        return hnew

    def smooth(self, weight=0.5):
        hnew = self.interpolate_nans()

        Zf = ndimage.filters.gaussian_filter(hnew.data,1,mode='nearest')
        hnew.data   = weight*Zf + (1.-weight)*hnew.data

        if hnew.uncert is not None:
            Uf = ndimage.filters.gaussian_filter(hnew.uncert,1,mode='nearest')
            hnew.uncert = weight*Uf + (1.-weight)*hnew.uncert

        return hnew

### slicing and shape changing
    def slices(self, axis=0):
        axes = []
        for i in range(self.dim):
            if i != axis:
                axes += [self.axes[i].clone()]

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

    def sum_over_axes(self, *axes):
        if len(axes) == self.dim:
            return self.sum()

        newaxes = []
        for i in range(self.dim):
            if i not in axes:
                newaxes += [self.axes[i].clone()]

        newdata = np.apply_over_axes(np.sum, self.data, axes)
        newdata.shape = tuple(n for n in newdata.shape if n > 1)

        if self.uncert is None:
            newuncert = None
        else:
            uncratio_sq = np.power(self.uncert / self.data, 2)
            uncratio_sqsum = np.apply_over_axes(np.sum, uncratio_sq, axes)
            uncratio_sqsum.shape = tuple(n for n in uncratio_sqsum.shape if n > 1)
            newuncert = np.sqrt(uncratio_sqsum) * newdata

        return Histogram(
            *newaxes,
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
        debug = kwargs.pop('debug',False)

        ### Range, validity and bounds checking
        assert(self.dim == 1)

        # only allow this many iterations when the minimization
        # routine fails and we do a random walk on the parameters
        maxcount = kwargs.pop('maxcount',20)
        test = kwargs.pop('test','chisquare').lower()
        uncert = kwargs.pop('uncert',self.uncert)

        sel = np.isfinite(self.grid) & np.isfinite(self.data)

        if uncert is None:
            xx = self.grid[sel].astype(np.float64)
            yy = self.data[sel].astype(np.float64)

        else:
            minuncert = kwargs.pop('minuncert',0)
            mindata = kwargs.pop('mindata',0.05)

            # only consider points where the uncertainty
            # is greater than minuncert
            sel &= uncert > minuncert

            # and only try to fit the data if at least mindata of the
            # bins have non-zero uncertainty
            ndatapoints = np.count_nonzero(sel)
            datasize = len(self.data)
            if (ndatapoints / datasize) < mindata:
                raise RuntimeError('not enough data to fit')

            ### Set up the data
            xx = self.grid[sel].astype(np.float64)
            yy = self.data[sel].astype(np.float64)

            if 'sigma' not in kwargs:
                kwargs['sigma'] = uncert[sel].astype(np.float64)
            if 'absolute_sigma' not in kwargs:
                kwargs['absolute_sigma'] = True

        # initial parameters
        if hasattr(p0, '__call__'):
            kwargs['p0'] = p0(self)
        else:
            kwargs['p0'] = copy(p0)

        npar = len(kwargs['p0'])

        ### Do the fit
        count = 0
        while count < maxcount:
            try:
                pfit,pcov = opt.curve_fit(fcn, xx, yy, **kwargs)

                if not isinstance(pcov,np.ndarray):
                    raise RuntimeError('bad fit')

                if debug:
                    print('init:',kwargs['p0'])
                    print('fit:',pfit)
                    print('pcov:',pcov)
                    print('kwargs:',kwargs)

                break
            except RuntimeError as e:
                if debug:
                    print('RuntimeError:',e)
                # random walk of the initial parameters
                # gradually increasing step size from 5% to 20%
                a = (float(count)/maxcount) * 0.15 + 0.05
                for i in range(npar):
                    kwargs['p0'][i] *= np.random.uniform(1.-a,1.+a)
                count = count + 1
            except TypeError as e:
                if debug:
                    print('TypeError:',e)
                raise RuntimeError('not enough data. TypeError: '+str(e))

        ### Make sure the fit converged
        if count == maxcount:
            # maybe try another fitting method here?
            raise RuntimeError('fit did not converge')

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
            if test[:nchar] == 'chisquare'[:nchar]:
                # simple Chi-squared test
                chisq,pval = stats.chisquare(yy,yyfit,len(pfit))
                ptest = (chisq/ndf,pval)
            elif test[:nchar] == 'kstest'[:nchar]:
                # two-sided Kolmogorov-Smirov test
                D,pval = stats.kstest(dyy,
                            stats.norm(0,dyy.std()).cdf)
                ptest = (D,pval)
            elif test[:nchar] == 'shapiro'[:nchar]:
                # Shapiro-Wilk test
                W,pval = sp.stats.shapiro(dyy)
                ptest = (W,pval)
            else:
                raise RuntimeError('unknown goodness of fit test')

            return pfit,pcov,ptest

        return pfit,pcov
