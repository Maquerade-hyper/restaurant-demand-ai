from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str

    holiday_multiplier: float = 1.0
    promotion_multiplier: float = 1.0
    event_multiplier: float = 1.0
    weather_multiplier: float = 1.0
    tourism_multiplier: float = 1.0
    demographic_multiplier: float = 1.0
    cultural_multiplier: float = 1.0

    min_multiplier: float = 0.50
    max_multiplier: float = 2.00

    metadata: Dict[str, str] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ScenarioResult:
    scenario_name: str
    baseline_demand: float
    scenario_demand: float
    absolute_change: float
    percentage_change: float