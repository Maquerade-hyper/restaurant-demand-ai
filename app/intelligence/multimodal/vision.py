from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


class VisualIntelligence:

    def __init__(
        self,
        image_size: int = 64,
    ):
        self.image_size = int(
            image_size
        )

    def _load_image(
        self,
        image_path: str,
    ):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for "
                "visual intelligence. "
                "Install with: "
                "pip install pillow"
            ) from exc

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        image = Image.open(
            path
        ).convert("RGB")

        return image

    def extract(
        self,
        image_path: str,
    ) -> Dict[str, float]:

        image = self._load_image(
            image_path
        )

        original_width, original_height = (
            image.size
        )

        image = image.resize(
            (
                self.image_size,
                self.image_size,
            )
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        r = array[:, :, 0]
        g = array[:, :, 1]
        b = array[:, :, 2]

        gray = (
            0.299 * r
            + 0.587 * g
            + 0.114 * b
        )

        features = {
            "visual_width": float(
                original_width
            ),
            "visual_height": float(
                original_height
            ),
            "visual_aspect_ratio": (
                float(original_width)
                / max(
                    float(original_height),
                    1.0,
                )
            ),
            "visual_mean_r": float(
                np.mean(r)
            ),
            "visual_mean_g": float(
                np.mean(g)
            ),
            "visual_mean_b": float(
                np.mean(b)
            ),
            "visual_std_r": float(
                np.std(r)
            ),
            "visual_std_g": float(
                np.std(g)
            ),
            "visual_std_b": float(
                np.std(b)
            ),
            "visual_mean_brightness": float(
                np.mean(gray)
            ),
            "visual_brightness_std": float(
                np.std(gray)
            ),
            "visual_contrast": float(
                np.percentile(gray, 90)
                - np.percentile(gray, 10)
            ),
        }

        # Saturation approximation.
        maximum = np.max(
            array,
            axis=2,
        )

        minimum = np.min(
            array,
            axis=2,
        )

        saturation = np.where(
            maximum == 0,
            0.0,
            (
                maximum - minimum
            )
            / np.maximum(
                maximum,
                1e-8,
            ),
        )

        features[
            "visual_mean_saturation"
        ] = float(
            np.mean(saturation)
        )

        features[
            "visual_saturation_std"
        ] = float(
            np.std(saturation)
        )

        # Edge-energy approximation.
        dx = np.diff(
            gray,
            axis=1,
        )

        dy = np.diff(
            gray,
            axis=0,
        )

        edge_energy = (
            np.mean(np.abs(dx))
            + np.mean(np.abs(dy))
        ) / 2.0

        features[
            "visual_edge_energy"
        ] = float(
            edge_energy
        )

        # Bright / dark pixel ratios.
        features[
            "visual_bright_ratio"
        ] = float(
            np.mean(gray > 0.75)
        )

        features[
            "visual_dark_ratio"
        ] = float(
            np.mean(gray < 0.25)
        )

        return features

    def quality(
        self,
        image_path: str,
    ) -> float:

        try:
            image = self._load_image(
                image_path
            )
        except Exception:
            return 0.0

        width, height = image.size

        resolution_score = min(
            (
                width * height
            )
            / float(
                512 * 512
            ),
            1.0,
        )

        dimension_score = (
            1.0
            if width >= 64
            and height >= 64
            else 0.5
        )

        return float(
            0.7 * resolution_score
            + 0.3 * dimension_score
        )