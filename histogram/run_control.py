from histogram.detail.run_control import RunControl, Bunch

### Run Control Parameters
rc = RunControl(
    overwrite = Bunch(
        overwrite = 'ask',
        timestamp = None,
        timeout   = 30*60,
    ),
    histdir = None,
    plot = Bunch(
        baseline = 'bottom',
        patch = Bunch(
            alpha = 0.6,
        ),
    ),
)
