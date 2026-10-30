"""
Regulatory Module

Reorganized into logical groups:
- capital: Capital adequacy and risk-weighted assets (Basel, IFRS9, SA-CCR)
- compliance: Liquidity and stress testing (LCR, NSFR, FRTB, stress tests)
"""

from .capital import *
from .compliance import *

# Combine __all__ from all submodules
from .capital import __all__ as _capital_all
from .compliance import __all__ as _compliance_all

__all__ = list(_capital_all) + list(_compliance_all)
