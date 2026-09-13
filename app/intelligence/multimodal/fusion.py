from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


class MultimodalFusion:

    DEFAULT_MODALITY_WEIGHTS = {
        "numeric": 0.50,
        "text": 0.25,
        "visual": 0.25,
    }

    def __init__(
        self,
        modality_weights=None,
    ):
        weights = dict(
            modality_weights
            or self.DEFAULT_MODALITY_WEIGHTS
        )

        for key, value in weights.items():

            value = float(value)

            if value < 0:
                raise ValueError(
                    "Modality weights "
                    "cannot be negative."
                )

            if not np.isfinite(value):
                raise ValueError(
                    "Modality weights "
                    "must be finite."
                )

            weights[key] = value

        total = sum(
            weights.values()
        )

        if total <= 0:
            raise ValueError(
                "At least one modality "
                "weight must be positive."
            )

        self.weights = {
            key: value / total
            for key, value in weights.items()
        }

    @staticmethod
    def _clean(
        features: Mapping[str, float],
    ) -> Dict[str, float]:

        cleaned = {}

        for key, value in features.items():

            value = float(value)

            if not np.isfinite(value):
                value = 0.0

            cleaned[str(key)] = value

        return cleaned

    def fuse(
        self,
        numeric_features=None,
        text_features=None,
        visual_features=None,
        text_quality=0.0,
        visual_quality=0.0,
    ):

        numeric_features = self._clean(
            numeric_features or {}
        )

        text_features = self._clean(
            text_features or {}
        )

        visual_features = self._clean(
            visual_features or {}
        )

        text_available = (
            len(text_features) > 0
        )

        visual_available = (
            len(visual_features) > 0
        )

        modality_count = (
            1
            + int(text_available)
            + int(visual_available)
        )

        features = {}

        # ----------------------------------------------------
        # NUMERIC
        # ----------------------------------------------------

        for key, value in (
            numeric_features.items()
        ):
            features[
                f"mm_numeric_{key}"
            ] = value

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        for key, value in (
            text_features.items()
        ):
            features[
                f"mm_text_{key}"
            ] = value

        # ----------------------------------------------------
        # VISUAL
        # ----------------------------------------------------

        for key, value in (
            visual_features.items()
        ):
            features[
                f"mm_visual_{key}"
            ] = value

        # ----------------------------------------------------
        # MODALITY PRESENCE
        # ----------------------------------------------------

        features[
            "mm_has_numeric"
        ] = 1.0

        features[
            "mm_has_text"
        ] = float(text_available)

        features[
            "mm_has_visual"
        ] = float(visual_available)

        features[
            "mm_modality_count"
        ] = float(modality_count)

        features[
            "mm_text_quality"
        ] = float(
            np.clip(
                text_quality,
                0.0,
                1.0,
            )
        )

        features[
            "mm_visual_quality"
        ] = float(
            np.clip(
                visual_quality,
                0.0,
                1.0,
            )
        )

        # ----------------------------------------------------
        # CROSS-MODAL INTERACTIONS
        # ----------------------------------------------------

        text_context = float(
            text_features.get(
                "text_context_density",
                0.0,
            )
        )

        text_promotion = float(
            text_features.get(
                "text_promotion_score",
                0.0,
            )
        )

        text_event = float(
            text_features.get(
                "text_event_score",
                0.0,
            )
        )

        text_weather = float(
            text_features.get(
                "text_weather_score",
                0.0,
            )
        )

        visual_brightness = float(
            visual_features.get(
                "visual_mean_brightness",
                0.0,
            )
        )

        visual_saturation = float(
            visual_features.get(
                "visual_mean_saturation",
                0.0,
            )
        )

        numeric_demand = float(
            numeric_features.get(
                "demand",
                numeric_features.get(
                    "forecast_demand",
                    0.0,
                ),
            )
        )

        # These are interaction signals,
        # not causal claims.
        features[
            "mm_text_numeric_interaction"
        ] = (
            text_context
            * numeric_demand
        )

        features[
            "mm_promotion_numeric_interaction"
        ] = (
            text_promotion
            * numeric_demand
        )

        features[
            "mm_event_numeric_interaction"
        ] = (
            text_event
            * numeric_demand
        )

        features[
            "mm_weather_numeric_interaction"
        ] = (
            text_weather
            * numeric_demand
        )

        features[
            "mm_text_visual_interaction"
        ] = (
            text_context
            * visual_brightness
        )

        features[
            "mm_promotion_visual_interaction"
        ] = (
            text_promotion
            * visual_saturation
        )

        # ----------------------------------------------------
        # FUSION SCORE
        # ----------------------------------------------------

        available_weight = (
            self.weights.get(
                "numeric",
                0.0,
            )
        )

        if text_available:
            available_weight += (
                self.weights.get(
                    "text",
                    0.0,
                )
            )

        if visual_available:
            available_weight += (
                self.weights.get(
                    "visual",
                    0.0,
                )
            )

        quality_components = [
            self.weights.get(
                "numeric",
                0.0,
            )
        ]

        if text_available:
            quality_components.append(
                self.weights.get(
                    "text",
                    0.0,
                )
                * float(
                    np.clip(
                        text_quality,
                        0.0,
                        1.0,
                    )
                )
            )

        if visual_available:
            quality_components.append(
                self.weights.get(
                    "visual",
                    0.0,
                )
                * float(
                    np.clip(
                        visual_quality,
                        0.0,
                        1.0,
                    )
                )
            )

        quality_score = (
            sum(quality_components)
            / max(
                available_weight,
                1e-12,
            )
        )

        fusion_score = (
            available_weight
            * quality_score
        )

        features[
            "mm_fusion_score"
        ] = float(
            np.clip(
                fusion_score,
                0.0,
                1.0,
            )
        )

        features[
            "mm_quality_score"
        ] = float(
            np.clip(
                quality_score,
                0.0,
                1.0,
            )
        )

        return features