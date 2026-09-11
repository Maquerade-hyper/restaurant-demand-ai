from datetime import date


SEED = 42

NUM_OUTLETS = 130

START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)

COUNTRIES = {
    "India": ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Pune"],
    "USA": ["New York", "Chicago", "Los Angeles", "Houston"],
    "UK": ["London", "Manchester", "Birmingham"],
    "UAE": ["Dubai", "Abu Dhabi", "Sharjah"],
    "Singapore": ["Singapore"],
    "Saudi Arabia": ["Riyadh", "Jeddah"],
}

OUTLET_TYPES = [
    "restaurant",
    "bar",
    "restaurant_bar",
    "cloud_kitchen",
    "delivery_kitchen",
]

LOCATION_TYPES = [
    "business",
    "residential",
    "tourist",
    "university",
    "mixed",
]