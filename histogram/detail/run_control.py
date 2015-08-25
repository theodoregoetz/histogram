class Singleton:
    def __init__(self,cls):
        self.cls = cls
        self.instance = None
    def __call__(self,*args,**kwargs):
        if self.instance is None:
            self.instance = self.cls(*args,**kwargs)
        return self.instance

class RunControlBunch(dict):
    def __init__(self,**kwargs):
        dict.__init__(self,kwargs)
        self.__dict__ = self

    def __getattr__(self,key):
        self.__dict__[key] = RunControlBunch()
        return self.__dict__[key]

    def flat_str(self, parent=None):
        ret = []
        for key,value in sorted(self.__dict__.items()):
            if parent is not None:
                parent_key = parent+'.'+key
            else:
                parent_key = key
            if isinstance(value,RunControlBunch):
                ret.append(value.flat_str(parent_key))
            else:
                if isinstance(value,str):
                    ret.append(parent_key+" = '"+str(value)+"'")
                else:
                    ret.append(parent_key+' = '+str(value))
        return '\n'.join(ret)

@Singleton
class RunControl(RunControlBunch):
    def __init__(self,**kwargs):
        RunControlBunch.__init__(self,**kwargs)

    def __str__(self):
        return self.flat_str()

    def lock(self):
        def locked_setattr(self,k,v):
            if k not in self.__dict__:
                raise KeyError('Run control parameters have been locked.')
            else:
                self.__dict__[k] = v
        RunControlBunch.__setattr__ = lambda self,k,v: locked_setattr(self,k,v)

    def unlock(self):
        RunControlBunch.__setattr__ = dict.__setattr__
