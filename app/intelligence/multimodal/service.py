from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .schemas import (
    TextContext,
)
from .text import TextIntelligence
from .vision import VisualIntelligence
from .fusion import MultimodalFusion


class MultimodalIntelligenceService:

    FORBIDDEN_COLUMNS = {
        "quantity_sold",
        "revenue",
        "true_demand",
        "deconstrained_demand",
        "actual_demand",
        "future_demand",
        "lost_demand",
        "prediction",
        "target",
    }

    def __init__(
        self,
        text_engine: Optional[
            TextIntelligence
        ] = None,
        visual_engine: Optional[
            VisualIntelligence
        ] = None,
        fusion_engine: Optional[
            MultimodalFusion
        ] = None,
    ):

        self.text_engine = (
            text_engine
            or TextIntelligence()
        )

        self.visual_engine = (
            visual_engine
            or VisualIntelligence()
        )

        self.fusion_engine = (
            fusion_engine
            or MultimodalFusion()
        )

    def process(
        self,
        numeric_features=None,
        text: Optional[str] = None,
        text_source: str = "unknown",
        image_path: Optional[str] = None,
        image_source: str = "unknown",
    ):

        numeric_features = dict(
            numeric_features or {}
        )

        # ----------------------------------------------------
        # Remove known target/leakage fields.
        # ----------------------------------------------------

        safe_numeric = {}

        for key, value in (
            numeric_features.items()
        ):

            if str(key) in (
                self.FORBIDDEN_COLUMNS
            ):
                continue

            try:
                value = float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if not np.isfinite(value):
                continue

            safe_numeric[str(key)] = value

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text_features = {}

        text_quality = 0.0

        if text is not None:

            context = TextContext(
                text=str(text),
                source=text_source,
            )

            text_features = (
                self.text_engine.extract(
                    context
                )
            )

            text_quality = (
                self.text_engine.quality(
                    context
                )
            )

        # ----------------------------------------------------
        # VISUAL
        # ----------------------------------------------------

        visual_features = {}

        visual_quality = 0.0

        if image_path is not None:

            visual_features = (
                self.visual_engine.extract(
                    image_path
                )
            )

            visual_quality = (
                self.visual_engine.quality(
                    image_path
                )
            )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        fused = self.fusion_engine.fuse(
            numeric_features=safe_numeric,
            text_features=text_features,
            visual_features=visual_features,
            text_quality=text_quality,
            visual_quality=visual_quality,
        )

        result = {
            "features": fused,
            "text_available": bool(
                text_features
            ),
            "visual_available": bool(
                visual_features
            ),
            "modality_count": (
                1
                + int(bool(text_features))
                + int(bool(visual_features))
            ),
            "text_quality": text_quality,
            "visual_quality": visual_quality,
        }

        return result

    def process_batch(
        self,
        rows,
    ):

        output = []

        for row in rows:

            result = self.process(
                numeric_features=row.get(
                    "numeric_features",
                    {},
                ),
                text=row.get(
                    "text"
                ),
                text_source=row.get(
                    "text_source",
                    "unknown",
                ),
                image_path=row.get(
                    "image_path"
                ),
                image_source=row.get(
                    "image_source",
                    "unknown",
                ),
            )

            flat = dict(
                result["features"]
            )

            flat[
                "text_available"
            ] = float(
                result["text_available"]
            )

            flat[
                "visual_available"
            ] = float(
                result["visual_available"]
            )

            flat[
                "modality_count"
            ] = float(
                result["modality_count"]
            )

            output.append(flat)

        return output

    def validate(
        self,
        result,
    ):

        errors = []

        if not isinstance(
            result,
            dict,
        ):
            return {
                "passed": False,
                "errors": [
                    "result must be dict"
                ],
            }

        if "features" not in result:
            errors.append(
                "missing features"
            )

        if "modality_count" not in result:
            errors.append(
                "missing modality_count"
            )

        if "text_available" not in result:
            errors.append(
                "missing text_available"
            )

        if "visual_available" not in result:
            errors.append(
                "missing visual_available"
            )

        features = result.get(
            "features",
            {},
        )

        for key, value in features.items():

            if not np.isfinite(
                float(value)
            ):
                errors.append(
                    f"non-finite feature: {key}"
                )

        for key in [
            "text_quality",
            "visual_quality",
        ]:

            if key in result:

                value = float(
                    result[key]
                )

                if not (
                    0.0
                    <= value
                    <= 1.0
                ):
                    errors.append(
                        f"{key} outside [0,1]"
                    )

        return {
            "passed": len(errors) == 0,
            "errors": errors,
        }