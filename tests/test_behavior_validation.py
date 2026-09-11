from app.data.generators.behavior_validation import (
    generate_behavior_sample,
    validate_behavior,
)


def test_behavior_sample():

    df = generate_behavior_sample(days=30)

    assert len(df) == 30


def test_behavior_relationships():

    df = generate_behavior_sample(days=30)

    checks = validate_behavior(df)

    assert all(checks.values())





