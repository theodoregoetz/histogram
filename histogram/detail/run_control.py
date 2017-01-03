class RunControlBunch(dict):
    '''
    This is the classic Python "Bunch" class to be used
    as a base-class (with a Singleton decorator) for
    holding run control parameters.
    '''
    def __init__(self,**kwargs):
        dict.__init__(self,kwargs)
        self.__dict__ = self

    def __getattr__(self,key):
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


class RunControl(RunControlBunch):
    '''
    A lockable bunch class to hold
    run control parameters.
    '''

    def __init__(self,**kwargs):
        RunControlBunch.__init__(self,**kwargs)

    def __str__(self):
        return self.flat_str()

    def lock(self):
        '''
        Prevents any new parameters from being created
        though it still allows already defined parameters
        to be changed.
        '''
        def locked_setattr(self,k,v):
            if k not in self.__dict__:
                raise KeyError('Unknown key. Run control parameters have been locked.')
            else:
                self.__dict__[k] = v
        RunControlBunch.__setattr__ = lambda self,k,v: locked_setattr(self,k,v)

    def unlock(self):
        '''
        Allows creation of new parameters.
        '''
        RunControlBunch.__setattr__ = dict.__setattr__

