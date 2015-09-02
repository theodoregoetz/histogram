from . import histogram

try:
    from .histogram_hdf5 import save_histograms, load_histograms
except ImportError:
    warn('Could not import h5py. You will not be able to load/save histograms stored in hdf5 format.', ImportWarning)
