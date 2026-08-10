import numpy as np
import pandas as pd

from appliance_energy import backtest
from appliance_energy.models import benchmarks


def _series(values, start="2020-01-01"):
    index = pd.date_range(start=start, periods=len(values), freq="h")
    return pd.Series(np.asarray(values, dtype=float), index=index)


def _split(n_train=200, n_test=72):
    full = _series(np.arange(n_train + n_test))
    return full.iloc[:n_train], full.iloc[n_train:]


def test_rolling_forecast_covers_whole_test_index():
    train, test = _split()

    result = backtest.rolling_origin_forecast(
        benchmarks.naive_forecast, train, test, horizon=24
    )

    assert len(result) == len(test)
    assert list(result.index) == list(test.index)


def test_rolling_forecast_handles_ragged_final_block():
    # 70 test points with horizon 24 gives blocks of 24, 24, 22.
    train, test = _split(n_test=70)

    result = backtest.rolling_origin_forecast(
        benchmarks.naive_forecast, train, test, horizon=24
    )

    assert len(result) == 70
    assert list(result.index) == list(test.index)


def test_rolling_naive_updates_origin_each_block():
    # On a strictly increasing series, a rolling naive forecast should
    # restart from the last revealed value at every block, rather than
    # repeating a single value across the whole test period.
    train, test = _split(n_train=100, n_test=48)

    result = backtest.rolling_origin_forecast(
        benchmarks.naive_forecast, train, test, horizon=24
    )

    # Block 1 repeats the final training value.
    assert (result.iloc[:24] == train.iloc[-1]).all()
    # Block 2 repeats the final value of block 1's realised data.
    assert (result.iloc[24:] == test.iloc[23]).all()


def test_rolling_forecast_never_uses_future_values():
    # A probe function records the length of history it was handed at
    # each origin; it must only ever grow by one horizon at a time and
    # never exceed the data available before the block being predicted.
    train, test = _split(n_train=100, n_test=72)
    seen_lengths = []

    def probe(history, horizon, index):
        seen_lengths.append(len(history))
        return pd.Series(0.0, index=index)

    backtest.rolling_origin_forecast(probe, train, test, horizon=24)

    assert seen_lengths == [100, 124, 148]


def test_rolling_seasonal_naive_matches_manual_blocks():
    train, test = _split(n_train=200, n_test=48)
    seasonality = 24

    result = backtest.rolling_origin_forecast(
        benchmarks.seasonal_naive_forecast, train, test,
        horizon=24, seasonality=seasonality,
    )

    # First block: the last 24 training values.
    assert np.allclose(result.iloc[:24].values, train.iloc[-seasonality:].values)
    # Second block: the first 24 test values, now revealed.
    assert np.allclose(result.iloc[24:].values, test.iloc[:seasonality].values)


def test_rolling_origin_indices_partition_the_test_set():
    _, test = _split(n_test=70)

    blocks = backtest.rolling_origin_indices(test, horizon=24)

    assert blocks == [(0, 24), (24, 48), (48, 70)]
    assert blocks[-1][1] == len(test)


def test_name_is_applied_when_given():
    train, test = _split()

    result = backtest.rolling_origin_forecast(
        benchmarks.naive_forecast, train, test, horizon=24, name="naive"
    )

    assert result.name == "naive"
