from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.intelligence.daily.engine import (
    DailyIntelligenceEngine,
)


ROOT = Path(__file__).resolve().parents[3]

OUTPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "daily_intelligence.json"
)


def _json_safe(value: Any) -> Any:
    """
    Convert NumPy/Pandas scalar values into native Python
    values before JSON serialization.
    """

    if isinstance(
        value,
        (
            np.integer,
            np.floating,
            np.bool_,
        ),
    ):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _json_safe(item)
            for item in value
        ]

    return value


class DailyIntelligenceService:
    def __init__(
        self,
        dataset_path: str | Path | None = None,
        output_path: str | Path = OUTPUT_PATH,
    ) -> None:
        self.engine = DailyIntelligenceEngine(
            dataset_path=(
                dataset_path
                if dataset_path is not None
                else (
                    ROOT
                    / "data"
                    / "interim"
                    / "demand_censoring_intelligence.csv"
                )
            )
        )

        self.output_path = Path(
            output_path
        )

    def generate(
        self,
        report_date: str | None = None,
        include_normal: bool = False,
        write_output: bool = True,
    ) -> dict:
        report = self.engine.generate(
            report_date=report_date,
            include_normal=include_normal,
        )

        report = _json_safe(report)

        if not report["validation"]["passed"]:
            raise ValueError(
                "Daily intelligence validation failed: "
                + str(
                    report["validation"]["errors"]
                )
            )

        if write_output:
            self.output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with self.output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    report,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

        return report

    def load_latest(self) -> dict:
        if not self.output_path.exists():
            raise FileNotFoundError(
                "Daily intelligence report not found: "
                f"{self.output_path}"
            )

        with self.output_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)