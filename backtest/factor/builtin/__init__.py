"""Built-in factor definitions packaged with the backtest framework.

Imports here trigger ``@register`` decorators on package load, populating
``data/factor_library/registry.json`` with the Barra style factors and any
other framework-level factors.
"""

from backtest.factor.builtin import barra  # noqa: F401
