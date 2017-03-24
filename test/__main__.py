from test import main

from .serialization.test_histogram_hist import *
from .serialization.test_histogram_hdf5 import *
from .serialization.test_histogram_numpy import *
from .serialization.test_histogram_root import *
from .test_histogram import *
from .test_histogram_axis import *
from .test_run_control import *

main()
