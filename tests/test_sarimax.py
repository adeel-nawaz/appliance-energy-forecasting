import numpy as np
import pandas as pd
import pytest

from appliance_energy.models import sarimax


@pytest.fixture(scope="module")
def seasonal_series():
    """A short, well-behaved seasonal series so the fits stay fast."""
    rng = np.random.RandomState(0)
    n = 24 * 21
    index = pd.date_range("2020-01-01", periods=n, freq="h")

    hours = np.arange(n) % 24
    daily_cycle = 50 + 30 * np.sin(2 * np.pi * hours / 24)
    noise = rng.normal(0, 5, n)

    return pd.Series(daily_cycle + noise, index=index)


def test_trend_is_constant_only_without_differencing():
    assert sarimax._trend_for_order((1, 0, 1), (0, 0, 0, 0)) == "c"
    assert sarimax._trend_for_order((1, 1, 1), (0, 0, 0, 0)) == "n"
    assert sarimax._trend_for_order((1, 0, 1), (1, 1, 1, 24)) == "n"


def test_grid_search_covers_every_combination(seasonal_series):
    p_range, d_range, q_range = range(0, 2), range(0, 2), range(0, 2)

    results = sarimax.grid_search_orders(
        seasonal_series, p_range=p_range, d_range=d_range, q_range=q_range,
        seasonal_order=None, maxiter=10, verbose=False,
    )

    assert len(results) == len(p_range) * len(d_range) * len(q_range)
    assert {"p", "d", "q", "aic", "bic", "converged"}.issubset(results.columns)


def test_grid_search_results_sorted_by_aic_ascending(seasonal_series):
    results = sarimax.grid_search_orders(
        seasonal_series, p_range=range(0, 2), d_range=range(0, 1),
        q_range=range(0, 2), seasonal_order=None, maxiter=10, verbose=False,
    )

    aic_values = results["aic"].dropna().values
    assert np.all(np.diff(aic_values) >= 0)


def test_grid_search_records_failures_without_raising(seasonal_series):
    # A deliberately extreme order on a very short series should be
    # recorded as an error row rather than aborting the whole search.
    short = seasonal_series.iloc[:30]

    results = sarimax.grid_search_orders(
        short, p_range=range(0, 1), d_range=range(0, 1), q_range=range(0, 1),
        seasonal_order=None, maxiter=5, verbose=False,
    )

    assert len(results) == 1
    assert "error" in results.columns


def test_best_order_from_results_extracts_best_converged_row(seasonal_series):
    results = sarimax.grid_search_orders(
        seasonal_series, p_range=range(0, 2), d_range=range(0, 1),
        q_range=range(0, 2), seasonal_order=None, maxiter=10, verbose=False,
    )

    order, seasonal_order = sarimax.best_order_from_results(results)

    # Selection is made among converged fits only, so the expected winner
    # is the lowest-AIC row that actually converged -- not necessarily the
    # lowest-AIC row overall.
    converged = results[results["converged"].astype(bool)].dropna(subset=["aic"])
    top = converged.sort_values("aic").iloc[0]

    assert order == (int(top["p"]), int(top["d"]), int(top["q"]))
    assert len(seasonal_order) == 4


def test_best_order_ignores_convergence_when_disabled(seasonal_series):
    results = sarimax.grid_search_orders(
        seasonal_series, p_range=range(0, 2), d_range=range(0, 1),
        q_range=range(0, 2), seasonal_order=None, maxiter=10, verbose=False,
    )

    order, _ = sarimax.best_order_from_results(results, require_converged=False)

    top = results.dropna(subset=["aic"]).sort_values("aic").iloc[0]
    assert order == (int(top["p"]), int(top["d"]), int(top["q"]))


def test_best_order_raises_when_nothing_converged():
    empty = pd.DataFrame({
        "p": [1], "d": [0], "q": [1], "P": [0], "D": [0], "Q": [0], "s": [0],
        "aic": [np.nan],
    })

    with pytest.raises(ValueError):
        sarimax.best_order_from_results(empty)


def test_best_order_skips_non_converged_models_with_lower_aic():
    # A failed optimisation can still report an AIC. The converged model
    # must win even though the failed one has a (meaningless) lower AIC.
    results = pd.DataFrame({
        "p": [4, 2], "d": [1, 0], "q": [1, 1],
        "P": [0, 0], "D": [0, 0], "Q": [0, 0], "s": [0, 0],
        "aic": [1.0, 500.0],
        "converged": [False, True],
    })

    order, _ = sarimax.best_order_from_results(results)

    assert order == (2, 0, 1)


def test_best_order_falls_back_when_nothing_converged_but_aic_exists():
    results = pd.DataFrame({
        "p": [1, 2], "d": [0, 0], "q": [1, 1],
        "P": [0, 0], "D": [0, 0], "Q": [0, 0], "s": [0, 0],
        "aic": [900.0, 100.0],
        "converged": [False, False],
    })

    order, _ = sarimax.best_order_from_results(results)

    assert order == (2, 0, 1)


def test_refine_seasonal_warns_when_d_varies_under_simple_differencing(seasonal_series):
    # Mixed d values mean different effective sample sizes, so the AIC
    # values would not be comparable; the caller must be warned.
    with pytest.warns(UserWarning, match="not comparable"):
        sarimax.refine_seasonal_orders(
            seasonal_series.iloc[:200],
            candidate_orders=[(1, 0, 0), (1, 1, 0)],
            seasonal_p_range=range(0, 1), seasonal_d_range=range(1, 2),
            seasonal_q_range=range(0, 1), period=24,
            maxiter=5, verbose=False, simple_differencing=True,
        )


def test_refine_seasonal_does_not_warn_when_d_is_constant(seasonal_series):
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)

        sarimax.refine_seasonal_orders(
            seasonal_series.iloc[:200],
            candidate_orders=[(1, 0, 0), (0, 0, 1)],
            seasonal_p_range=range(0, 1), seasonal_d_range=range(1, 2),
            seasonal_q_range=range(0, 1), period=24,
            maxiter=5, verbose=False, simple_differencing=True,
        )


def test_forecast_length_and_interval_ordering(seasonal_series):
    train = seasonal_series.iloc[:-24]
    horizon = 24
    index = seasonal_series.index[-24:]

    fit = sarimax.fit_sarimax(train, order=(1, 0, 0), seasonal_order=None, maxiter=20)
    forecast = sarimax.forecast_sarimax(fit, horizon=horizon, index=index, alpha=0.05)

    assert len(forecast) == horizon
    assert list(forecast.index) == list(index)
    assert {"sarimax", "lower", "upper"}.issubset(forecast.columns)

    # The interval must bracket the point forecast at every step.
    assert (forecast["lower"] <= forecast["sarimax"]).all()
    assert (forecast["sarimax"] <= forecast["upper"]).all()


def test_wider_alpha_gives_narrower_interval(seasonal_series):
    train = seasonal_series.iloc[:-24]
    index = seasonal_series.index[-24:]

    fit = sarimax.fit_sarimax(train, order=(1, 0, 0), seasonal_order=None, maxiter=20)

    ci_95 = sarimax.forecast_sarimax(fit, 24, index, alpha=0.05)
    ci_80 = sarimax.forecast_sarimax(fit, 24, index, alpha=0.20)

    width_95 = (ci_95["upper"] - ci_95["lower"]).mean()
    width_80 = (ci_80["upper"] - ci_80["lower"]).mean()

    assert width_80 < width_95


def test_residual_diagnostics_returns_expected_keys(seasonal_series):
    fit = sarimax.fit_sarimax(seasonal_series, order=(1, 0, 0), seasonal_order=None, maxiter=20)

    diagnostics = sarimax.residual_diagnostics(fit, lags=24)

    expected = {
        "n_resid", "mean", "std", "skew", "kurtosis",
        "ljung_box_stat", "ljung_box_pvalue", "residuals_uncorrelated",
        "jarque_bera_stat", "jarque_bera_pvalue", "residuals_normal",
    }
    assert expected.issubset(diagnostics.keys())


def test_save_and_load_best_params_roundtrip(tmp_path):
    path = tmp_path / "best_params.json"

    sarimax.save_best_params((2, 1, 3), (1, 1, 0, 24), extra={"aic": 123.4}, path=path)
    order, seasonal_order = sarimax.load_best_params(path=path)

    assert order == (2, 1, 3)
    assert seasonal_order == (1, 1, 0, 24)


def test_load_best_params_falls_back_when_missing(tmp_path):
    from appliance_energy import config

    order, seasonal_order = sarimax.load_best_params(path=tmp_path / "does_not_exist.json")

    assert order == config.SARIMAX_ORDER
    assert seasonal_order == config.SARIMAX_SEASONAL_ORDER
