from collections import deque

class skippable:
    def __init__(self,it): self.it = iter(it)
    def __iter__(self): return self
    def __next__(self): return self,next(self.it)
    def next(self): return self.__next__()
    def __call__(self,n):
        for _ in range(n):
            next(self.it,None)

def window(seq,size=2,wintype=tuple):
    assert size > 1
    it = iter(seq)
    items = deque((next(it,None) for _ in range(size)), maxlen=size)
    yield wintype(items)
    for item in it:
        items.append(item)
        yield wintype(items) if wintype else items
    for _ in range(size-1):
        items.append(None)
        yield wintype(items) if wintype else items

