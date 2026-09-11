import random

import pandas as pd

from app.data.generators.config import (
    COUNTRIES,
    LOCATION_TYPES,
    NUM_OUTLETS,
    OUTLET_TYPES,
)


CUISINES = [
    "Indian",
    "Italian",
    "American",
    "Chinese",
    "Mexican",
    "Arabic",
    "International",
]

KITCHENS = [
    "standard",
    "central",
    "cloud",
    "delivery",
]


def generate_outlets(seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)

    rows = []

    locations = [
        (country, city)
        for country, cities in COUNTRIES.items()
        for city in cities
    ]

    for i in range(1, NUM_OUTLETS + 1):
        country, city = rng.choice(locations)
        outlet_type = rng.choice(OUTLET_TYPES)
        location_type = rng.choice(LOCATION_TYPES)

        rows.append(
            {
                "outlet_id": f"O{i:03d}",
                "outlet_type": outlet_type,
                "country": country,
                "region": country,
                "city": city,
                "latitude": round(rng.uniform(-35, 55), 6),
                "longitude": round(rng.uniform(-120, 150), 6),
                "location_type": location_type,
                "cuisine": rng.choice(CUISINES),
                "capacity": rng.randint(30, 300),
                "delivery_available": outlet_type
                in {"cloud_kitchen", "delivery_kitchen", "restaurant", "restaurant_bar"},
                "takeaway_available": rng.random() < 0.75,
                "bar_available": outlet_type in {"bar", "restaurant_bar"},
                "kitchen_type": rng.choice(KITCHENS),
                "tourism_level": round(rng.random(), 3),
                "business_area": round(rng.random(), 3),
                "residential_area": round(rng.random(), 3),
                "student_area": round(rng.random(), 3),
            }
        )

    return pd.DataFrame(rows)