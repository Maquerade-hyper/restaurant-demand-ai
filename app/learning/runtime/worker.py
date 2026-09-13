from __future__ import annotations

import logging
import time

from .runtime_service import LearningRuntimeService


class LearningRuntimeWorker:

    def __init__(
        self,
        service: LearningRuntimeService,
        interval_seconds: int = 60,
    ):
        self.service = service
        self.interval_seconds = max(
            5,
            int(interval_seconds),
        )

        self.logger = logging.getLogger(
            "restaurant_demand_ai.learning"
        )

    def run_forever(self) -> None:

        self.logger.info(
            "Continuous learning runtime started."
        )

        self.logger.info(
            "Training source: REAL CLIENT DATA ONLY."
        )

        self.logger.info(
            "Synthetic continuous training: DISABLED."
        )

        self.logger.info(
            "Scan interval: %s seconds",
            self.interval_seconds,
        )

        while True:

            try:
                results = (
                    self.service.scan_once()
                )

                if not results:
                    self.logger.info(
                        self.service.last_message
                        or "No client data detected. Waiting."
                    )

            except Exception:
                self.logger.exception(
                    "Learning runtime scan failed."
                )

            time.sleep(
                self.interval_seconds
            )