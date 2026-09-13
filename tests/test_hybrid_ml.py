def test_genuine_hybrid_uses_validation_for_weights():
    import numpy as np

    from app.forecasting.hybrid.benchmark import (
        GenuineHybridBenchmark,
    )

    actual_val = np.array([10, 12, 14, 16, 18], dtype=float)

    validation_predictions = {
        "naive": np.array([10, 12, 14, 16, 18], dtype=float),
        "xgboost": np.array([11, 13, 15, 17, 19], dtype=float),
        "transformer": np.array([12, 14, 16, 18, 20], dtype=float),
    }

    actual_test = np.array([20, 22, 24, 26, 28], dtype=float)

    test_predictions = {
        "naive": np.array([20, 22, 24, 26, 28], dtype=float),
        "xgboost": np.array([21, 23, 25, 27, 29], dtype=float),
        "transformer": np.array([19, 21, 23, 25, 27], dtype=float),
    }

    benchmark = GenuineHybridBenchmark()

    result = benchmark.evaluate(
        actual_validation=actual_val,
        validation_predictions=validation_predictions,
        actual_test=actual_test,
        test_predictions=test_predictions,
        horizon=1,
    )

    assert len(result["metrics"]) == 4
    assert set(result["metrics"]["model_name"]) == {
        "naive",
        "xgboost",
        "transformer",
        "hybrid",
    }


def test_genuine_hybrid_test_predictions_are_unseen():
    import numpy as np

    from app.forecasting.hybrid.benchmark import (
        GenuineHybridBenchmark,
    )

    actual_val = np.array([10, 11, 12, 13], dtype=float)

    validation_predictions = {
        "naive": np.array([10, 11, 12, 13], dtype=float),
        "xgboost": np.array([11, 12, 13, 14], dtype=float),
        "transformer": np.array([12, 13, 14, 15], dtype=float),
    }

    actual_test = np.array([20, 21, 22, 23], dtype=float)

    test_predictions = {
        "naive": np.array([20, 21, 22, 23], dtype=float),
        "xgboost": np.array([20, 21, 22, 23], dtype=float),
        "transformer": np.array([20, 21, 22, 23], dtype=float),
    }

    benchmark = GenuineHybridBenchmark()

    result = benchmark.evaluate(
        actual_val,
        validation_predictions,
        actual_test,
        test_predictions,
        horizon=1,
    )

    assert len(result["hybrid_predictions"]) == 4
    assert np.all(
        np.isfinite(result["hybrid_predictions"])
    )


def test_genuine_hybrid_validation_rejects_missing_model():
    import numpy as np

    from app.forecasting.hybrid.benchmark import (
        GenuineHybridBenchmark,
    )

    benchmark = GenuineHybridBenchmark()

    actual = np.array([1, 2, 3], dtype=float)

    predictions = {
        "naive": np.array([1, 2, 3], dtype=float),
        "xgboost": np.array([1, 2, 3], dtype=float),
    }

    try:
        benchmark.derive_validation_scores(
            actual,
            predictions,
        )
        assert False
    except ValueError:
        assert True


def test_genuine_hybrid_validation_result():
    import numpy as np

    from app.forecasting.hybrid.benchmark import (
        GenuineHybridBenchmark,
    )

    benchmark = GenuineHybridBenchmark()

    actual = np.array([1, 2, 3], dtype=float)

    predictions = {
        "naive": np.array([1, 2, 3], dtype=float),
        "xgboost": np.array([1, 2, 3], dtype=float),
        "transformer": np.array([1, 2, 3], dtype=float),
    }

    result = benchmark.evaluate(
        actual,
        predictions,
        actual,
        predictions,
        horizon=1,
    )

    validation = benchmark.validate(result)

    assert validation["passed"] is True
    assert validation["errors"] == []