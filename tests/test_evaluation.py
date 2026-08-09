import numpy as np
import pandas as pd
import pytest

from appliance_energy import evaluation


def _make_series(values, start="2020-01-01", freq="h"):
    index = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=index, dtype=float)


def test_mase_zero_for_perfect_forecast():
    y_train = _make_series(np.sin(np.linspace(0, 20, 200)) + 10)
    y_test = _make_series([5.0, 6.0, 7.0], start=y_train.index[-1] + pd.Timedelta(hours=1))

    result = evaluation.mase(y_test, y_test, y_train, seasonality=24)

    assert result == 0.0


def test_mase_scales_by_training_seasonal_naive_error():
    # MASE is defined so that a forecast whose error equals the average
    # in-sample seasonal-naive error scores exactly 1.
    y_train = _make_series(np.arange(100, dtype=float))
    seasonality = 24

    naive_errors = np.abs(
        y_train.iloc[seasonality:].values - y_train.iloc[:-seasonality].values
    )
    scale = naive_errors.mean()

    y_true = _make_series([scale, scale, scale])
    y_pred = _make_series([0.0, 0.0, 0.0], start=y_true.index[0])

    result = evaluation.mase(y_true, y_pred, y_train, seasonality=seasonality)

    assert result == pytest.approx(1.0)


def test_mae_rmse_bias_simple_case():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    assert evaluation.mae(y_true, y_pred) == pytest.approx((2 + 2 + 3) / 3)
    assert evaluation.bias(y_true, y_pred) == pytest.approx((2 - 2 + 3) / 3)
    assert evaluation.rmse(y_true, y_pred) >= evaluation.mae(y_true, y_pred)


def test_evaluate_forecast_returns_expected_keys():
    y_train = _make_series(np.random.RandomState(0).rand(100) * 100)
    y_true = _make_series([50.0, 55.0], start=y_train.index[-1] + pd.Timedelta(hours=1))
    y_pred = _make_series([51.0, 53.0], start=y_true.index[0])

    result = evaluation.evaluate_forecast("dummy_model", y_true, y_pred, y_train, seasonality=24)

    assert set(result.keys()) == {"model", "MAE", "RMSE", "MASE", "Bias"}
    assert result["model"] == "dummy_model"


def test_evaluate_all_sorts_by_mase_ascending():
    y_train = _make_series(np.random.RandomState(0).rand(200) * 100)
    y_test = _make_series(
        np.random.RandomState(1).rand(24) * 100,
        start=y_train.index[-1] + pd.Timedelta(hours=1),
    )

    forecasts = {
        "perfect": y_test.copy(),
        "way_off": y_test + 1000,
    }

    results = evaluation.evaluate_all(forecasts, y_test, y_train, seasonality=24)

    assert list(results["model"]) == ["perfect", "way_off"]
    assert results.loc[0, "MASE"] < results.loc[1, "MASE"]
