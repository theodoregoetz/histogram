from warnings import warn

try:
    from .graphics import plothist
    have_graphics = True
except ImportError:
    warn('Could not import graphics.', ImportWarning)
    have_graphics = False
