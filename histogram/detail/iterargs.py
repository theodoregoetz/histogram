from collections import deque

class skippable:
    def __init__(self,it): self.it = iter(it)
    def __iter__(self): return self
    def __next__(self): return self,next(self.it)
    def __call__(self,n):
        for _ in range(n):
            next(self.it)

def window(seq,size=2,wintype=tuple):
    i = iter(seq)
    items = deque((next(i,None) for _ in range(size)), maxlen=size)
    while items[0] is not None:
        yield wintype(items)
        items.append(next(i,None))
