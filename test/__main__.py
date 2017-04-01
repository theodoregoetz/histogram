from test import main

from .graphics.test_histogram_mpl import *
from .serialization.test_ask_overwrite import *
from .serialization.test_histogram_hist import *
from .serialization.test_histogram_hdf5 import *
from .serialization.test_histogram_numpy import *
from .serialization.test_histogram_root import *
from .serialization.test_serialization import *
from .test_histogram import *
from .test_histogram_axis import *
from .test_run_control import *

main()
