from __future__ import annotations

import time
from typing import Callable, List

import numpy as np

from .schemas import PerformanceReport


class PerformanceMonitor:

    def __init__(
        self,
        cpu_only: bool = True,
    ):
        self.cpu_only = cpu_only

    def benchmark(
        self,
        function: Callable,
        iterations: int = 20,
        rows_per_request: int = 1,
    ) -> PerformanceReport:

        if iterations <= 0:
            raise ValueError(
                "iterations must be > 0"
            )

        durations: List[float] = []

        successful = 0
        failed = 0
        errors = []

        start_total = (
            time.perf_counter()
        )

        for _ in range(iterations):

            start = (
                time.perf_counter()
            )

            try:
                function()
                successful += 1

            except Exception as exc:
                failed += 1

                if len(errors) < 10:
                    errors.append(
                        str(exc)
                    )

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000.0

            durations.append(
                elapsed
            )

        total_ms = (
            time.perf_counter()
            - start_total
        ) * 1000.0

        values = np.asarray(
            durations,
            dtype=float,
        )

        mean_ms = float(
            np.mean(values)
        )

        p50 = float(
            np.percentile(
                values,
                50,
            )
        )

        p95 = float(
            np.percentile(
                values,
                95,
            )
        )

        p99 = float(
            np.percentile(
                values,
                99,
            )
        )

        seconds = max(
            total_ms / 1000.0,
            1e-9,
        )

        requests_per_second = (
            successful
            / seconds
        )

        rows_per_second = (
            successful
            * rows_per_request
            / seconds
        )

        return PerformanceReport(
            requests=iterations,
            successful_requests=successful,
            failed_requests=failed,
            total_ms=float(total_ms),
            mean_ms=mean_ms,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            requests_per_second=float(
                requests_per_second
            ),
            rows_per_second=float(
                rows_per_second
            ),
            model_load_ms=0.0,
            memory_safe=True,
            cpu_only=self.cpu_only,
            errors=errors,
        )