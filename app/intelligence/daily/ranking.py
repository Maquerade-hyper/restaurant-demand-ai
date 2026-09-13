from __future__ import annotations


PRIORITY_ORDER = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MODERATE": 2,
    "LOW": 3,
    "NORMAL": 4,
}


def priority_value(priority: str) -> int:
    return PRIORITY_ORDER.get(str(priority).upper(), 99)


def rank_actions(actions: list[dict]) -> list[dict]:
    return sorted(
        actions,
        key=lambda item: (
            priority_value(item.get("priority", "NORMAL")),
            -float(
                item.get("inventory", {}).get(
                    "shortage",
                    0,
                )
            ),
            -float(
                item.get("forecast", {}).get(
                    "d7",
                    0,
                )
            ),
        ),
    )


def summarize_priorities(actions: list[dict]) -> dict:
    result = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MODERATE": 0,
        "LOW": 0,
        "NORMAL": 0,
    }

    for action in actions:
        priority = str(
            action.get("priority", "NORMAL")
        ).upper()

        if priority not in result:
            priority = "NORMAL"

        result[priority] += 1

    return result