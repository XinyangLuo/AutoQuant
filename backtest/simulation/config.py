from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class SimulationConfig:
    """回测模拟配置。"""

    initial_cash: float = 100_000_000.0
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.001
    transfer_fee_rate: float = 0.00002
    price_type: Literal["o2o", "c2c"] = "o2o"
    allow_short: bool = True
    benchmark: str | None = None
