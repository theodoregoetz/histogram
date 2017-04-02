from .detail import RunControl

### Run Control Parameters (singleton object)
rc = RunControl()

rc.fill_type = 'int'
rc.plot.baseline = 'bottom'
rc.plot.patch.alpha = 0.6

# prevent new parameters from being created
# (the values can still be changed)
rc.lock()
