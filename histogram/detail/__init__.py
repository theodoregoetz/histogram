from .run_control import RunControl
from .strings import isstr, encode_dict, encoded_str, decode_dict, \
                     decoded_str

def isinteger(i):
    if not hasattr(isinteger,'_int_types'):
        try:
            isinteger._int_types = (int,long)
        except NameError:
            isinteger._int_types = (int,)
    return isinstance(i, isinteger._int_types)
