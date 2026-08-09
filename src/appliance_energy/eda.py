"""
Exploratory time-series diagnostics: seasonal decomposition and
stationarity testing (ADF, differencing).
"""

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def decompose_series(series, period, model="additive"):
    """Classical seasonal decomposition into trend, seasonal, and residual."""
    return seasonal_decompose(series.dropna(), period=period, model=model)


def adf_test(series, name="series", alpha=0.05, verbose=True):
    """
    Augmented Dickey-Fuller test for a unit root.

    Returns a dict with the test statistic, p-value, critical values,
    and a boolean stationarity verdict at the given significance level.
    A low p-value (< alpha) rejects the null hypothesis of a unit root,
    i.e. the series is judged stationary.
    """

    series = series.dropna()
    stat, pvalue, used_lag, n_obs, crit_values, _ = adfuller(series, autolag="AIC")

    is_stationary = pvalue < alpha

    result = {
        "name": name,
        "adf_statistic": stat,
        "p_value": pvalue,
        "used_lag": used_lag,
        "n_obs": n_obs,
        "critical_values": crit_values,
        "is_stationary": is_stationary,
    }

    if verbose:
        verdict = "stationary" if is_stationary else "non-stationary"
        print(
            f"ADF test on '{name}': statistic={stat:.4f}, "
            f"p-value={pvalue:.4f} -> {verdict} at alpha={alpha}"
        )

    return result


def difference_series(series, periods=1):
    """First-order (or seasonal, via `periods`) differencing."""
    return series.diff(periods).dropna()


def stationarity_report(series, seasonal_period=24, alpha=0.05):
    """
    Run ADF tests on the raw series, its first difference, and its
    seasonal difference. Returns a summary dataframe, one row per test.
    """

    rows = [
        adf_test(series, name="level", alpha=alpha),
        adf_test(difference_series(series, 1), name="first_difference", alpha=alpha),
        adf_test(
            difference_series(series, seasonal_period),
            name=f"seasonal_difference_{seasonal_period}",
            alpha=alpha,
        ),
    ]

    return pd.DataFrame(rows)[
        ["name", "adf_statistic", "p_value", "used_lag", "n_obs", "is_stationary"]
    ]
