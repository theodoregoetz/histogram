import math
from copy import copy
import numpy as np

from matplotlib import pyplot, cm, _cm
from matplotlib.colors import LinearSegmentedColormap

def generate_cmap_flame():
    clrs = [
        (1,1,1),
        (0,0.3,1),
        (0,1,1),
        (0,1,0.3),
        (1,1,0),
        (1,0.5,0),
        (1,0,0),
        (0.5,0,0)
        ]
    flame = LinearSegmentedColormap.from_list('flame', clrs)
    flame.set_bad(flame(0)) # set nan's and inf's to white
    return flame

flame = generate_cmap_flame()
pyplot.register_cmap('flame',cmap=flame)

#cubehelix = LinearSegmentedColormap('cubehelix',cm.revcmap(_cm.cubehelix(1,0,1,2.5)))
#cubehelix.set_bad(cubehelix(0))

def cmap_powerlaw_adjust(cmap, a):
    '''
    returns a new colormap based on the one given
    but adjusted via power-law:

    newcmap = oldcmap**a
    '''
    if a < 0.:
        return cmap
    cdict = copy(cmap._segmentdata)
    def fn(x):
        return (x[0]**a, x[1], x[2])
    for key in ('red','green','blue'):
        try:
            cdict[key] = map(fn, cdict[key])
            cdict[key].sort()
            assert (cdict[key][0]<0 or cdict[key][-1]>1), \
                "Resulting indices extend out of the [0, 1] segment."
        except TypeError:
            def fngen(f):
                def fn(x):
                    return f(x)**a
                return fn
            cdict[key] = fngen(cdict[key])
    newcmap = LinearSegmentedColormap('colormap',cdict,1024)
    newcmap.set_bad(cmap(np.nan))
    return newcmap

def cmap_center_adjust(cmap, center_ratio):
    '''
    returns a new colormap based on the one given
    but adjusted so that the old center point is higher
    (>0.5) or lower (<0.5)
    '''
    if not (0. < center_ratio) & (center_ratio < 1.):
        return cmap
    a = math.log(center_ratio) / math.log(0.5)
    return cmap_powerlaw_adjust(cmap, a)

def cmap_center_point_adjust(cmap, range, center):
    '''
    converts center to a ratio between 0 and 1 of the
    range given and calls cmap_center_adjust(). returns
    a new adjusted colormap accordingly
    '''
    if not (range[0] < center < range[1]):
        return cmap
    center_ratio = (float(center) - range[0]) / (range[1] - range[0])
    return cmap_center_adjust(cmap, center_ratio)

if __name__ == "__main__":
    from matplotlib import pyplot, cm
    a = np.linspace(0, 1, 256).reshape(1,-1)
    a = np.vstack((a,a))

    flame_high = cmap_center_point_adjust(flame, [0,1], 0.75)
    jet_low = cmap_center_point_adjust(cm.jet, [-100,100], -75)

    fig = pyplot.figure(figsize=(6,5))
    for i,cm in enumerate([flame, cm.jet, flame_high, jet_low]):
        ax = fig.add_subplot(2,2,i+1)
        ax.imshow(a, cmap=cm, aspect='auto', origin='lower')

    pyplot.show()
