from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    model_name: str

    mae: float
    rmse: float
    smape: float
    bias: float

    high_demand_mae: float
    high_demand_bias: float
    spike_recall: float

    stability: float
    prediction_cost: float

    score: float = 0.0