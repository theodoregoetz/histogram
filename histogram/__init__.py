from warnings import warn

from .run_control import rc

from .histogram_axis import HistogramAxis
from .histogram import Histogram

from .serialization import *

try:
    from .graphics import *
except ImportError:
    warn('Could not import matplotlib. Proceeding without graphics...', ImportWarning)
