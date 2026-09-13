from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ProductionForecastRequest:
    outlet_id: str
    product_id: str
    horizon: int = 1


@dataclass
class ProductionForecastResponse:
    outlet_id: str
    product_id: str
    horizon: int
    predictions: List[float]
    model_name: str
    model_version: str
    inference_ms: float


@dataclass
class PerformanceReport:
    requests: int
    successful_requests: int
    failed_requests: int
    total_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    rows_per_second: float
    model_load_ms: float
    memory_safe: bool
    cpu_only: bool
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "requests": self.requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_ms": self.total_ms,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "requests_per_second": self.requests_per_second,
            "rows_per_second": self.rows_per_second,
            "model_load_ms": self.model_load_ms,
            "memory_safe": self.memory_safe,
            "cpu_only": self.cpu_only,
            "errors": self.errors,
        }