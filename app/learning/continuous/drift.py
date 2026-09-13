from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from .schemas import DriftReport


class DriftMonitor:
    """
    Lightweight CPU-first drift monitor.

    PSI:
        Population Stability Index.

    Interpretation:
        < 0.10  -> low/no meaningful drift
        0.10-0.25 -> moderate drift
        > 0.25  -> significant drift

    These are monitoring thresholds, not claims about model quality.
    """

    def __init__(
        self,
        psi_threshold: float = 0.25,
        mean_shift_threshold: float = 0.20,
        bins: int = 10,
    ):
        self.psi_threshold = float(psi_threshold)
        self.mean_shift_threshold = float(mean_shift_threshold)
        self.bins = int(bins)

    @staticmethod
    def _clean(values) -> np.ndarray:
        arr = pd.to_numeric(
            pd.Series(values),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(
            dtype=float
        )

        return arr

    def _psi(self, reference, current) -> float:
        ref = self._clean(reference)
        cur = self._clean(current)

        if len(ref) == 0 or len(cur) == 0:
            return 0.0

        combined = np.concatenate([ref, cur])

        if np.allclose(combined, combined[0]):
            return 0.0

        edges = np.quantile(
            combined,
            np.linspace(0.0, 1.0, self.bins + 1),
        )

        edges = np.unique(edges)

        if len(edges) < 2:
            return 0.0

        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)

        ref_pct = ref_counts.astype(float) / max(len(ref), 1)
        cur_pct = cur_counts.astype(float) / max(len(cur), 1)

        eps = 1e-6

        ref_pct = np.clip(ref_pct, eps, None)
        cur_pct = np.clip(cur_pct, eps, None)

        psi = np.sum(
            (cur_pct - ref_pct)
            * np.log(cur_pct / ref_pct)
        )

        return float(max(psi, 0.0))

    def compare_feature(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature: str,
    ) -> DriftReport:
        ref = self._clean(reference)
        cur = self._clean(current)

        ref_mean = float(np.mean(ref)) if len(ref) else 0.0
        cur_mean = float(np.mean(cur)) if len(cur) else 0.0

        denominator = max(abs(ref_mean), 1e-6)

        mean_shift = abs(cur_mean - ref_mean) / denominator
        psi = self._psi(ref, cur)

        drifted = (
            psi >= self.psi_threshold
            or mean_shift >= self.mean_shift_threshold
        )

        return DriftReport(
            feature=feature,
            reference_mean=ref_mean,
            current_mean=cur_mean,
            mean_shift=float(mean_shift),
            psi=float(psi),
            drifted=bool(drifted),
        )

    def compare(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        features: Iterable[str],
    ) -> List[DriftReport]:
        reports = []

        for feature in features:
            if feature not in reference.columns:
                continue

            if feature not in current.columns:
                continue

            reports.append(
                self.compare_feature(
                    reference[feature],
                    current[feature],
                    feature,
                )
            )

        return reports

    @staticmethod
    def error_degradation(
        reference_actual,
        reference_prediction,
        current_actual,
        current_prediction,
    ) -> float:
        """
        Relative MAE degradation.

        Positive value means current error is worse.
        Negative means current error improved.
        """

        ref_actual = np.asarray(reference_actual, dtype=float)
        ref_pred = np.asarray(reference_prediction, dtype=float)

        cur_actual = np.asarray(current_actual, dtype=float)
        cur_pred = np.asarray(current_prediction, dtype=float)

        ref_mae = float(np.mean(np.abs(ref_actual - ref_pred)))
        cur_mae = float(np.mean(np.abs(cur_actual - cur_pred)))

        if ref_mae <= 1e-12:
            return 0.0 if cur_mae <= 1e-12 else 1.0

        return float((cur_mae - ref_mae) / ref_mae)