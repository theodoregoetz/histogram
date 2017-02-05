class Singleton:
    '''
    Singleton decorator for classes
    '''
    def __init__(self, cls):
        self.cls = cls
        self.instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.cls(*args, **kwargs)
        return self.instance


class RunControlBunch(dict):
    '''
    This is the classic Python "Bunch" class to be used
    as a base-class (with a Singleton decorator) for
    holding run control parameters.
    '''
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

    def __getattr__(self, key):
        self.__dict__[key] = RunControlBunch()
        return self.__dict__[key]

    def flat_str(self, parent=None):
        '''
        Prints all parameters in the format:
            key = value
            key.subkey = value
            ...
        '''
        ret = []
        for key, value in sorted(self.__dict__.items()):
            if parent is not None:
                parent_key = parent+'.'+key
            else:
                parent_key = key
            if isinstance(value, RunControlBunch):
                ret.append(value.flat_str(parent_key))
            else:
                if isinstance(value, str):
                    ret.append(parent_key+" = '"+str(value)+"'")
                else:
                    ret.append(parent_key+' = '+str(value))
        return '\n'.join(ret)


@Singleton
class RunControl(RunControlBunch):
    '''
    A generic singleton bunch class to hold global
    run control parameters.
    '''
    def __init__(self, **kwargs):
        RunControlBunch.__init__(self, **kwargs)

    def __str__(self):
        return self.flat_str()

    def lock(self):
        '''
        Prevents any new parameters from being created
        though it still allows already defined parameters
        to be changed.
        '''
        def locked_setattr(self, k, v):
            if k in self.__dict__:
                self.__dict__[k] = v
            else:
                msg = 'Unknown key. Run control parameters have been locked.'
                raise KeyError(msg)
        RunControlBunch.__setattr__ = locked_setattr

    def unlock(self):
        '''
        Allows creation of new parameters.
        '''
        RunControlBunch.__setattr__ = dict.__setattr__
