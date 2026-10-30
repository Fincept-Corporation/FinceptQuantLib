"""
Financial Ratio Feature Engineering

ML-friendly wrappers around the ``finceptquantlib.analysis.fundamentals``
ratio functions. Computes a standard set of ~20 financial ratios
as a flat dictionary suitable for use as model features.

If the analysis module is not available (e.g., during isolated testing),
functions degrade gracefully and return empty dictionaries.

Functions
---------
- compute_ratio_features: Compute standard financial ratio features
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def compute_ratio_features(
    income: Any,
    balance: Any,
    cash_flow: Any,
    prior_balance: Optional[Any] = None,
) -> Dict[str, float]:
    """Compute a standard set of financial ratios as ML features.

    Wraps ratio functions from ``finceptquantlib.analysis.fundamentals`` to
    produce a flat dictionary of ~20 commonly used financial ratios. If the
    analysis module is unavailable, returns an empty dict gracefully.

    Parameters
    ----------
    income : IncomeStatement
        Income statement data (from
        ``finceptquantlib.analysis.fundamentals.statements``).
    balance : BalanceSheet
        Balance sheet data.
    cash_flow : CashFlowStatement
        Cash flow statement data.
    prior_balance : BalanceSheet, optional
        Prior period balance sheet for turnover ratios. If not provided,
        turnover-based ratios are omitted.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping ratio names to float values. Keys are
        lowercase with underscores. Returns empty dict if the
        analysis module cannot be imported.

    Examples
    --------
    >>> from finceptquantlib.analysis.fundamentals.statements import (
    ...     IncomeStatement, BalanceSheet, CashFlowStatement
    ... )
    >>> inc = IncomeStatement(revenue=1000.0, gross_profit=400.0,
    ...     operating_income=200.0, net_income=150.0)
    >>> bal = BalanceSheet(total_assets=5000.0, total_equity=2000.0)
    >>> cf = CashFlowStatement(net_income=150.0, depreciation_amortization=100.0)
    >>> ratios = compute_ratio_features(inc, bal, cf)
    >>> ratios["net_margin"]
    0.15

    Notes
    -----
    The function catches all import and computation errors per-ratio
    so that partial results are returned even if some ratios fail.
    """
    features: Dict[str, float] = {}

    try:
        from finceptquantlib.analysis.fundamentals import profitability as prof
        from finceptquantlib.analysis.fundamentals import liquidity as liq
        from finceptquantlib.analysis.fundamentals import solvency as solv
        from finceptquantlib.analysis.fundamentals import efficiency as eff
        from finceptquantlib.analysis.fundamentals import cashflow as cf_mod
    except ImportError:
        return features

    # -- Profitability ratios --
    _safe_add(features, "gross_margin", prof.gross_margin, income)
    _safe_add(features, "operating_margin", prof.operating_margin, income)
    _safe_add(features, "ebitda_margin", prof.ebitda_margin, income)
    _safe_add(features, "net_margin", prof.net_margin, income)
    _safe_add(
        features, "return_on_assets",
        prof.return_on_assets, income, balance,
    )
    _safe_add(
        features, "return_on_equity",
        prof.return_on_equity, income, balance,
    )
    _safe_add(
        features, "return_on_invested_capital",
        prof.return_on_invested_capital, income, balance,
    )
    _safe_add(
        features, "return_on_capital_employed",
        prof.return_on_capital_employed, income, balance,
    )

    # -- Liquidity ratios --
    _safe_add(features, "current_ratio", liq.current_ratio, balance)
    _safe_add(features, "quick_ratio", liq.quick_ratio, balance)
    _safe_add(features, "cash_ratio", liq.cash_ratio, balance)

    # -- Solvency ratios --
    _safe_add(features, "debt_to_equity", solv.debt_to_equity, balance)
    _safe_add(features, "debt_to_assets", solv.debt_to_assets, balance)
    _safe_add(
        features, "interest_coverage",
        solv.interest_coverage, income,
    )
    _safe_add(
        features, "financial_leverage",
        solv.financial_leverage, balance,
    )

    # -- Efficiency ratios (require prior balance for averages) --
    if prior_balance is not None:
        _safe_add(
            features, "asset_turnover",
            eff.asset_turnover, income, balance, prior_balance,
        )
        _safe_add(
            features, "inventory_turnover",
            eff.inventory_turnover, income, balance, prior_balance,
        )
        _safe_add(
            features, "receivables_turnover",
            eff.receivables_turnover, income, balance, prior_balance,
        )

    # -- Cash flow ratios --
    _safe_add(
        features, "free_cash_flow",
        cf_mod.free_cash_flow, cash_flow,
    )
    _safe_add(
        features, "cash_flow_margin",
        cf_mod.operating_cash_flow_margin, cash_flow, income,
    )
    _safe_add(
        features, "quality_of_earnings",
        cf_mod.quality_of_earnings, cash_flow, income,
    )

    return features


def _safe_add(
    features: Dict[str, float],
    name: str,
    func: Any,
    *args: Any,
) -> None:
    """Safely call a ratio function and add to features dict.

    Parameters
    ----------
    features : Dict[str, float]
        Target dictionary to add the result to.
    name : str
        Feature name key.
    func : callable
        Ratio function to call.
    *args : Any
        Arguments to pass to the function.
    """
    try:
        val = func(*args)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            features[name] = float(val)
    except Exception:
        pass
