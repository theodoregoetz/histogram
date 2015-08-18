class Bunch(object):
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

    def flat_str(self, parent=None):
        ret = []
        for key,value in sorted(self.__dict__.items()):
            if parent is not None:
                parent_key = parent+'.'+key
            else:
                parent_key = key
            if isinstance(value,Bunch):
                ret.append(value.flat_str(parent_key))
            else:
                if isinstance(value,str):
                    ret.append(parent_key+" = '"+str(value)+"'")
                else:
                    ret.append(parent_key+' = '+str(value))
        return '\n'.join(ret)

class Singleton:
    def __init__(self,cls):
        self.cls = cls
        self.instance = None
    def __call__(self,*args,**kwargs):
        if self.instance is None:
            self.instance = self.cls(*args,**kwargs)
        return self.instance

@Singleton
class RunControl(Bunch):
    def __init__(self,**kwargs):
        Bunch.__init__(self,**kwargs)

    def __str__(self):
        return self.flat_str()
