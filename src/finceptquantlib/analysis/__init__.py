"""
Financial Analysis Module

Reorganized into logical groups:
- fundamentals: Core financial ratios and statement analysis
- valuation: Valuation methodologies (DCF, residual income, etc.)
- industry: Industry-specific analysis (banking, insurance, REITs, utilities)
"""

from .fundamentals import *
from .valuation import *
from .industry import *

# Combine __all__ from all submodules
from .fundamentals import __all__ as _fundamentals_all
from .valuation import __all__ as _valuation_all
from .industry import __all__ as _industry_all

__all__ = list(dict.fromkeys(list(_fundamentals_all) + list(_valuation_all) + list(_industry_all)))
