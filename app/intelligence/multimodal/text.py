from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List

import numpy as np

from .schemas import TextContext


class TextIntelligence:

    SIGNAL_TERMS = {
        "promotion": {
            "discount",
            "offer",
            "deal",
            "promo",
            "promotion",
            "coupon",
            "sale",
            "special",
            "combo",
        },
        "holiday": {
            "holiday",
            "festival",
            "festive",
            "celebration",
            "vacation",
            "public holiday",
        },
        "event": {
            "event",
            "concert",
            "match",
            "game",
            "conference",
            "wedding",
            "party",
            "festival",
        },
        "weather": {
            "rain",
            "rainy",
            "storm",
            "snow",
            "heat",
            "hot",
            "cold",
            "temperature",
            "weather",
        },
        "delivery": {
            "delivery",
            "deliver",
            "online",
            "app",
            "takeaway",
            "takeout",
        },
        "dine_in": {
            "restaurant",
            "dining",
            "dine",
            "dine-in",
            "table",
            "reservation",
        },
        "premium": {
            "premium",
            "luxury",
            "signature",
            "exclusive",
            "gourmet",
        },
        "vegetarian": {
            "vegetarian",
            "vegan",
            "plant-based",
            "veg",
        },
        "non_vegetarian": {
            "chicken",
            "beef",
            "lamb",
            "mutton",
            "fish",
            "seafood",
            "meat",
            "non-veg",
        },
        "beverage": {
            "coffee",
            "tea",
            "drink",
            "beverage",
            "juice",
            "bar",
            "cocktail",
            "mocktail",
        },
        "family": {
            "family",
            "kids",
            "children",
            "group",
        },
        "student": {
            "student",
            "university",
            "college",
            "campus",
        },
        "tourism": {
            "tourist",
            "tourism",
            "hotel",
            "visitor",
            "travel",
        },
        "urgency": {
            "limited",
            "today",
            "tonight",
            "last",
            "urgent",
            "ending",
        },
    }

    def __init__(
        self,
        vocabulary: Iterable[str] | None = None,
    ):
        self.vocabulary = set(
            vocabulary or []
        )

    @staticmethod
    def _normalize(text: str) -> str:
        text = str(text).lower()
        text = re.sub(
            r"[^a-z0-9\s\-]",
            " ",
            text,
        )
        text = re.sub(
            r"\s+",
            " ",
            text,
        )
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        normalized = self._normalize(text)

        if not normalized:
            return []

        return normalized.split()

    def extract(
        self,
        context: TextContext,
    ) -> Dict[str, float]:

        tokens = self.tokenize(
            context.text
        )

        counts = Counter(tokens)

        token_count = max(
            len(tokens),
            1,
        )

        features = {
            "text_token_count": float(
                len(tokens)
            ),
            "text_unique_token_count": float(
                len(set(tokens))
            ),
            "text_length": float(
                len(context.text)
            ),
            "text_vocabulary_density": (
                len(set(tokens))
                / token_count
            ),
        }

        for signal_name, terms in (
            self.SIGNAL_TERMS.items()
        ):

            count = sum(
                counts.get(term, 0)
                for term in terms
            )

            frequency = (
                count
                / token_count
            )

            features[
                f"text_{signal_name}_count"
            ] = float(count)

            features[
                f"text_{signal_name}_score"
            ] = float(
                min(
                    frequency * 10.0,
                    1.0,
                )
            )

            features[
                f"text_has_{signal_name}"
            ] = float(
                count > 0
            )

        # Sentiment-neutral intensity.
        # This intentionally avoids claiming semantic sentiment.
        intensity_terms = {
            "very",
            "highly",
            "major",
            "huge",
            "massive",
            "exclusive",
            "limited",
            "special",
        }

        intensity_count = sum(
            counts.get(term, 0)
            for term in intensity_terms
        )

        features[
            "text_intensity"
        ] = float(
            min(
                intensity_count
                / token_count
                * 10.0,
                1.0,
            )
        )

        # Simple context concentration.
        active_signals = sum(
            features[
                f"text_has_{name}"
            ]
            for name in self.SIGNAL_TERMS
        )

        features[
            "text_context_count"
        ] = float(
            active_signals
        )

        features[
            "text_context_density"
        ] = float(
            min(
                active_signals
                / max(
                    len(
                        self.SIGNAL_TERMS
                    ),
                    1,
                ),
                1.0,
            )
        )

        return features

    def quality(
        self,
        context: TextContext,
    ) -> float:

        text = str(
            context.text or ""
        ).strip()

        if not text:
            return 0.0

        tokens = self.tokenize(
            text
        )

        if not tokens:
            return 0.0

        length_score = min(
            len(tokens) / 20.0,
            1.0,
        )

        diversity_score = (
            len(set(tokens))
            / max(len(tokens), 1)
        )

        return float(
            min(
                0.6 * length_score
                + 0.4 * diversity_score,
                1.0,
            )
        )

    def batch_extract(
        self,
        contexts: Iterable[TextContext],
    ):
        rows = []

        for context in contexts:

            features = self.extract(
                context
            )

            features[
                "text_quality"
            ] = self.quality(
                context
            )

            features[
                "text_available"
            ] = 1.0

            rows.append(features)

        return rows