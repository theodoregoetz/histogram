from .run_control import RunControl
from .strings import isstr, encode_dict, encoded_str, decode_dict, \
                     decoded_str

try:
    long  # attempt to evaluate long
    def isinteger(i):
        return isinstance(i, (int,long))
except NameError:
    def isinteger(i):
        return isinstance(i, int)
