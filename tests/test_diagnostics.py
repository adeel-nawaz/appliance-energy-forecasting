import numpy as np
import pandas as pd
import pytest

from appliance_energy import evaluation


def _series(values, start="2020-01-01"):
    index = pd.date_range(start=start, periods=len(values), freq="h")
    return pd.Series(np.asarray(values, dtype=float), index=index)


def _daily_cycle(n, rng, noise=5.0):
    """A clean 24-hour sine cycle plus noise."""
    hours = np.arange(n) % 24
    return 50 + 30 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, noise, n)


@pytest.fixture
def scenario():
    """Test period of three days with a few models of differing quality."""
    rng = np.random.RandomState(0)

    train = _series(_daily_cycle(24 * 30, rng))
    test = _series(
        _daily_cycle(72, rng),
        start=train.index[-1] + pd.Timedelta(hours=1),
    )

    forecasts = {
        "good_model": test + rng.normal(0, 2, len(test)),
        "seasonal_naive_weekly": test + rng.normal(0, 10, len(test)),
        "mean": _series(np.full(len(test), train.mean()), start=test.index[0]),
        "naive": _series(np.full(len(test), train.iloc[-1]), start=test.index[0]),
    }

    return train, test, forecasts


# ------------------------------------------------------------------
# Strongest benchmark
# ------------------------------------------------------------------

def test_strongest_benchmark_picks_lowest_mase_among_benchmarks():
    results = pd.DataFrame({
        "model": ["sarimax", "seasonal_naive_weekly", "mean", "naive"],
        "MASE": [0.60, 0.81, 0.94, 1.60],
    })

    best = evaluation.strongest_benchmark(results)

    # sarimax has the lowest MASE overall but is not a benchmark.
    assert best["model"] == "seasonal_naive_weekly"


def test_strongest_benchmark_raises_when_no_benchmarks_present():
    results = pd.DataFrame({"model": ["sarimax", "feature_model"], "MASE": [0.6, 0.7]})

    with pytest.raises(ValueError, match="No benchmark models"):
        evaluation.strongest_benchmark(results)


# ------------------------------------------------------------------
# Skill scores
# ------------------------------------------------------------------

def test_skill_score_is_zero_for_the_baseline_itself():
    results = pd.DataFrame({
        "model": ["sarimax", "seasonal_naive_weekly"],
        "MAE": [40.0, 50.0], "RMSE": [80.0, 100.0], "MASE": [0.8, 1.0],
    })

    skill = evaluation.skill_scores(results, baseline_model="seasonal_naive_weekly")
    baseline_row = skill[skill["model"] == "seasonal_naive_weekly"].iloc[0]

    assert baseline_row["MASE_improvement_%"] == pytest.approx(0.0)
    assert not baseline_row["beats_benchmark"]


def test_skill_score_sign_and_magnitude():
    results = pd.DataFrame({
        "model": ["better", "worse", "seasonal_naive_weekly"],
        "MAE": [40.0, 60.0, 50.0],
        "RMSE": [80.0, 120.0, 100.0],
        "MASE": [0.8, 1.2, 1.0],
    })

    skill = evaluation.skill_scores(results, baseline_model="seasonal_naive_weekly")
    by_model = skill.set_index("model")

    assert by_model.loc["better", "MASE_improvement_%"] == pytest.approx(20.0)
    assert by_model.loc["worse", "MASE_improvement_%"] == pytest.approx(-20.0)
    assert by_model.loc["better", "beats_benchmark"]
    assert not by_model.loc["worse", "beats_benchmark"]


def test_mae_and_mase_improvements_agree(scenario):
    # MASE is MAE divided by a scale that is identical across models, so
    # percentage improvements on the two metrics must coincide.
    train, test, forecasts = scenario

    results = evaluation.evaluate_all(forecasts, test, train, seasonality=24)
    skill = evaluation.skill_scores(results)

    assert np.allclose(
        skill["MAE_improvement_%"], skill["MASE_improvement_%"], atol=1e-9
    )


def test_skill_scores_default_to_the_strongest_benchmark(scenario):
    train, test, forecasts = scenario

    results = evaluation.evaluate_all(forecasts, test, train, seasonality=24)
    skill = evaluation.skill_scores(results)

    assert skill.attrs["baseline_model"] == evaluation.strongest_benchmark(results)["model"]


# ------------------------------------------------------------------
# Error decomposition
# ------------------------------------------------------------------

def test_error_frame_is_signed_forecast_minus_actual(scenario):
    _, test, forecasts = scenario

    errors = evaluation.error_frame(forecasts, test)

    assert list(errors.columns) == list(forecasts)
    assert len(errors) == len(test)

    expected = forecasts["mean"] - test
    assert np.allclose(errors["mean"].values, expected.values)


def test_error_by_hour_has_one_row_per_hour(scenario):
    _, test, forecasts = scenario

    hourly = evaluation.error_by_hour(forecasts, test)

    assert len(hourly) == 24
    assert list(hourly.index) == list(range(24))
    assert (hourly >= 0).all().all()  # absolute by default


def test_error_by_step_ahead_matches_block_structure(scenario):
    _, test, forecasts = scenario

    steps = evaluation.error_by_step_ahead(forecasts, test, horizon=24)

    assert len(steps) == 24
    assert steps.index.name == "step_ahead"
    assert list(steps.index) == list(range(1, 25))


def test_step_ahead_is_confounded_with_hour_when_blocks_align(scenario):
    # When the test period is a whole number of aligned 24-hour blocks,
    # step position and hour of day carry identical information. This
    # pins that property so the diagnostic is not misread as isolating
    # forecast distance.
    _, test, forecasts = scenario

    by_hour = evaluation.error_by_hour(forecasts, test)
    by_step = evaluation.error_by_step_ahead(forecasts, test, horizon=24)

    first_hour = test.index[0].hour
    model = "good_model"

    reordered = [by_hour.loc[(first_hour + s - 1) % 24, model] for s in range(1, 25)]

    assert np.allclose(by_step[model].values, reordered)


def test_error_by_step_ahead_handles_ragged_final_block():
    train = _series(np.arange(200, dtype=float))
    test = _series(np.arange(30, dtype=float), start=train.index[-1] + pd.Timedelta(hours=1))
    forecasts = {"m": test + 1.0}

    steps = evaluation.error_by_step_ahead(forecasts, test, horizon=24)

    # 30 points over a 24-step horizon: steps 1-6 seen twice, 7-24 once.
    assert len(steps) == 24


def test_error_summary_percentiles_are_ordered(scenario):
    _, test, forecasts = scenario

    summary = evaluation.error_summary(forecasts, test)

    assert list(summary.columns) == ["median", "p75", "p90", "p95", "max"]

    for _, row in summary.iterrows():
        assert row["median"] <= row["p75"] <= row["p90"] <= row["p95"] <= row["max"]


def test_error_summary_rewards_the_accurate_model(scenario):
    _, test, forecasts = scenario

    summary = evaluation.error_summary(forecasts, test)

    assert summary.loc["good_model", "median"] < summary.loc["naive", "median"]


def test_residual_autocorrelation_shape_and_range(scenario):
    _, test, forecasts = scenario

    autocorr = evaluation.residual_autocorrelation(forecasts, test, lags=(1, 24))

    assert list(autocorr.columns) == ["lag_1", "lag_24"]
    assert set(autocorr.index) == set(forecasts)

    finite = autocorr.values[np.isfinite(autocorr.values)]
    assert np.all(np.abs(finite) <= 1.0 + 1e-9)


def test_constant_forecast_error_tracks_the_series(scenario):
    # A constant forecast leaves the target's own autocorrelation in the
    # error, so lag-1 autocorrelation should be clearly positive.
    _, test, forecasts = scenario

    autocorr = evaluation.residual_autocorrelation({"mean": forecasts["mean"]}, test,
                                                    lags=(1,))

    assert autocorr.loc["mean", "lag_1"] > 0.3
