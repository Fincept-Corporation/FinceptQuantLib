"""
Core Module

Reorganized into logical groups:
- foundations: Fundamental types, conventions, and automatic differentiation
- instruments: Period definitions and leg structures
- utilities: Result types, math utilities, and distributions
"""

from .foundations import *
from .instruments import *
from .utilities import *

# Combine __all__ from all submodules
from .foundations import __all__ as _foundations_all
from .instruments import __all__ as _instruments_all
from .utilities import __all__ as _utilities_all

__all__ = list(dict.fromkeys(list(_foundations_all) + list(_instruments_all) + list(_utilities_all)))
