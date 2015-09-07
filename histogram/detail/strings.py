import sys
from copy import copy
import codecs

try:
    basestring  # attempt to evaluate basestring
    def isstr(s):
        return isinstance(s, basestring)
except NameError:
    def isstr(s):
        return isinstance(s, str) or isinstance(s, bytes)

def encode_str(s):
    if s is None:
        return None
    if sys.version_info < (3,0):
        if not isinstance(s,unicode):
            s = unicode(s,'utf-8')
    elif isinstance(s, bytes):
        s = str(s,'utf-8')
    s = codecs.encode(s,'unicode-escape')
    return codecs.decode(s, 'latin-1')

def encode_dict(data):
    for k in data:
        if isstr(data[k]):
            data[k] = encode_str(data[k])
    return data

def decode_str(s):
    if s is None:
        return None
    return codecs.decode(s, 'unicode-escape')

def decode_dict(data):
    for k in data:
        if isstr(data[k]):
            data[k] = decode_str(data[k])
    return data
