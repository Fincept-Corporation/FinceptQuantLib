"""
Equilibrium Theory

General equilibrium and market equilibrium models:

Exchange Economy:
- ExchangeEconomy: Pure exchange model
- Consumer: Economic agent with preferences
- find_walrasian_equilibrium: Tatonnement algorithm
- contract_curve: Pareto efficient allocations

Consumer Types:
- CobbDouglasConsumer: CD preferences
- CESConsumer: Constant elasticity of substitution

Example:
    >>> from finceptquantlib.economics.equilibrium import (
    ...     ExchangeEconomy, CobbDouglasConsumer, find_walrasian_equilibrium
    ... )
    >>>
    >>> # 2-consumer, 2-good exchange economy
    >>> c1 = CobbDouglasConsumer(alphas=[0.6, 0.4], endowment=[10, 0])
    >>> c2 = CobbDouglasConsumer(alphas=[0.4, 0.6], endowment=[0, 10])
    >>>
    >>> economy = ExchangeEconomy(
    ...     consumers=[
    ...         Consumer(0, c1.endowment, c1.utility, c1.demand),
    ...         Consumer(1, c2.endowment, c2.utility, c2.demand),
    ...     ],
    ...     n_goods=2
    ... )
    >>>
    >>> prices, excess = find_walrasian_equilibrium(economy)
"""

from finceptquantlib.economics.equilibrium.general import (
    Consumer,
    ExchangeEconomy,
    find_walrasian_equilibrium,
    contract_curve,
    CobbDouglasConsumer,
    CESConsumer,
)

__all__ = [
    "Consumer",
    "ExchangeEconomy",
    "find_walrasian_equilibrium",
    "contract_curve",
    "CobbDouglasConsumer",
    "CESConsumer",
]
