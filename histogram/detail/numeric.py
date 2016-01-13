try:
    long  # attempt to evaluate long
    def isinteger(i):
        return isinstance(i, (int,long))
except NameError:
    def isinteger(i):
        return isinstance(i, int)
