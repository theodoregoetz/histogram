import os
import numpy as np
from cycler import cycler
from numpy import random as rand
from matplotlib import pyplot, cm

from histogram import Histogram

from matplotlib import rc as mplrc
from histogram import rc as histrc

mplrc('patch',facecolor='steelblue')
mplrc('axes',
    prop_cycle=cycler('color', ['steelblue','olive','darkred','goldenrod','salmon','plum','grey']),
    grid = True,
    facecolor = 'white', # axes background color
)
mplrc('axes.formatter', limits=(-3,4))
mplrc('image',cmap='cubehelix_r')

histrc.plot.baseline = 'left'

if not os.path.exists('test_images'):
    os.mkdir('test_images')

def test_plot_hist1d():
    npoints = 100000
    h1 = Histogram(100,(0,10),'x','y','title')
    h1.fill(rand.normal(5,2,npoints))

    fig,ax = pyplot.subplots(2,2)
    ax[0,0].plothist(h1, style='polygon' , baseline='bottom')
    ax[0,1].plothist(h1, style='errorbar', baseline='bottom')
    ax[1,0].plothist(h1, style='polygon' )#, baseline='left')
    ax[1,1].plothist(h1, style='errorbar')#, baseline='left')

    pyplot.savefig('test_images/test_plotting_fig_hist1d.png')


def test_plot_hist2d():
    npoints = 100000
    h2 = Histogram((100,(0,10),'x'),(100,(0,10),'y'),'z','title')
    h2.fill(rand.normal(5,2,npoints),
            rand.uniform(0,10,npoints))
    fig,ax = pyplot.subplots(1,2)
    ax[0].plothist(h2)
    ax[1].plothist(h2)
    ax[1].plothist(h2.smooth(1), style='contour', overlay=True)

    pyplot.savefig('test_images/test_plotting_fig_hist2d.png')
