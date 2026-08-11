import numpy as np
import pandas as pd
import pytest

from appliance_energy import config
from appliance_energy.models import foundation


class FakeChronosPipeline:
    """
    Stand-in for a Chronos pipeline.

    Returns a deterministic band around the mean of the context so the
    plumbing (shapes, indexing, quantile ordering, context handling) can
    be tested without downloading ~50M parameters of weights.
    """

    def __init__(self):
        self.calls = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        import torch

        context = inputs[0].numpy()
        self.calls.append(context.copy())

        centre = float(np.mean(context))
        offsets = np.array(quantile_levels) - 0.5

        # Shape (batch, prediction_length, n_quantiles), increasing in quantile.
        values = np.tile(
            centre + offsets * 100.0, (prediction_length, 1)
        )[None, ...]

        return torch.tensor(values, dtype=torch.float32), None


@pytest.fixture
def series():
    index = pd.date_range("2020-01-01", periods=24 * 30, freq="h")
    return pd.Series(np.arange(len(index), dtype=float), index=index)


@pytest.fixture
def split(series):
    return series.iloc[:-48], series.iloc[-48:]


# ------------------------------------------------------------------
# Single forecast
# ------------------------------------------------------------------

def test_forecast_returns_expected_shape_and_index(split):
    train, test = split
    pipeline = FakeChronosPipeline()

    result = foundation.forecast_chronos(
        pipeline, context=train.values, horizon=24, index=test.index[:24]
    )

    assert len(result) == 24
    assert list(result.index) == list(test.index[:24])
    assert {"foundation_model", "lower", "upper"}.issubset(result.columns)


def test_interval_brackets_the_point_forecast(split):
    train, test = split
    pipeline = FakeChronosPipeline()

    result = foundation.forecast_chronos(
        pipeline, context=train.values, horizon=24, index=test.index[:24]
    )

    assert (result["lower"] <= result["foundation_model"]).all()
    assert (result["foundation_model"] <= result["upper"]).all()


def test_context_is_truncated_to_the_model_window(split):
    train, test = split
    pipeline = FakeChronosPipeline()

    foundation.forecast_chronos(
        pipeline, context=train.values, horizon=24, index=test.index[:24],
        max_context=100,
    )

    assert len(pipeline.calls[0]) == 100
    # The retained window must be the most recent observations.
    assert np.allclose(pipeline.calls[0], train.values[-100:])


def test_short_context_is_passed_through_unpadded(split):
    train, test = split
    pipeline = FakeChronosPipeline()

    foundation.forecast_chronos(
        pipeline, context=train.values[:50], horizon=24, index=test.index[:24],
        max_context=2048,
    )

    assert len(pipeline.calls[0]) == 50


def test_default_quantiles_are_within_the_trained_range():
    # Chronos-Bolt is only trained on 0.1 ... 0.9; asking outside that
    # range is silently clipped, which would mislabel the interval.
    assert min(foundation.DEFAULT_QUANTILES) >= 0.1
    assert max(foundation.DEFAULT_QUANTILES) <= 0.9
    assert 0.5 in foundation.DEFAULT_QUANTILES


# ------------------------------------------------------------------
# Rolling forecast
# ------------------------------------------------------------------

def test_rolling_forecast_covers_the_whole_test_period(split):
    train, test = split
    pipeline = FakeChronosPipeline()

    result = foundation.rolling_forecast_chronos(
        pipeline, train, test, horizon=24
    )

    assert len(result) == len(test)
    assert list(result.index) == list(test.index)


def test_rolling_forecast_grows_context_by_one_block_each_origin(split):
    train, test = split
    pipeline = FakeChronosPipeline()

    foundation.rolling_forecast_chronos(pipeline, train, test, horizon=24)

    lengths = [len(c) for c in pipeline.calls]

    assert lengths == [len(train), len(train) + 24]


def test_rolling_forecast_never_sees_the_block_it_predicts(split):
    train, test = split
    pipeline = FakeChronosPipeline()

    foundation.rolling_forecast_chronos(pipeline, train, test, horizon=24)

    # The context for the first block must end at the last training value,
    # and must not contain any test observation.
    first_context = pipeline.calls[0]
    assert first_context[-1] == train.iloc[-1]
    assert not np.isin(test.values, first_context).any()

    # The second block's context may include block one, but nothing later.
    second_context = pipeline.calls[1]
    assert second_context[-1] == test.iloc[23]
    assert not np.isin(test.values[24:], second_context).any()


def test_rolling_forecast_handles_a_ragged_final_block(series):
    train, test = series.iloc[:-30], series.iloc[-30:]
    pipeline = FakeChronosPipeline()

    result = foundation.rolling_forecast_chronos(
        pipeline, train, test, horizon=24
    )

    assert len(result) == 30
    assert list(result.index) == list(test.index)


# ------------------------------------------------------------------
# Availability handling
# ------------------------------------------------------------------

def test_missing_dependency_raises_a_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "chronos":
            raise ImportError("no chronos here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(foundation.ChronosUnavailable, match="not installed"):
        foundation.load_chronos()
