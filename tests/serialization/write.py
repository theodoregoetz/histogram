import platform
from data import h

pyver = platform.python_version().split('.')[0]

h.save('h'+pyver)
h.save('h'+pyver+'.hdf5')
h.save('h'+pyver+'.root')
