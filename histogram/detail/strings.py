try:
    string_types = (basestring,)  # attempt to evaluate basestring
except NameError:
    string_types = (str,) #, bytes)
