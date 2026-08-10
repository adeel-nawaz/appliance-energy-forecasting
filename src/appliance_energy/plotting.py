"""Shared plotting utilities for time-series EDA and forecast comparison."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_time_series(series, title="", ylabel="", ax=None, **kwargs):
    """Plot single time series, optionally onto an existing axis."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.figure

    series.plot(ax=ax, **kwargs)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")

    fig.tight_layout()

    return fig, ax


def plot_seasonal_decomposition(result, title="Seasonal decomposition"):
    """Plot statsmodels DecomposeResult (observed/trend/seasonal/resid)."""

    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    return fig


def plot_acf_pacf(series, lags=48, title_prefix=""):
    """Plot ACF and PACF side by side (stacked) for a series."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"{title_prefix} ACF".strip())

    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"{title_prefix} PACF".strip())

    fig.tight_layout()

    return fig


def plot_forecasts(train, test, forecast_df, title="Appliance energy forecasting",
                    train_tail=14 * 24):
    """Plot recent training data, test data, and every forecast column."""

    fig, ax = plt.subplots(figsize=(14, 7))

    train.tail(train_tail).plot(ax=ax, label="Training data", linewidth=1.5)
    test.plot(ax=ax, label="Test data", linewidth=2.0, color="black")

    for col in forecast_df.columns:
        if col != "actual":
            forecast_df[col].plot(ax=ax, label=col, alpha=0.9)

    ax.set_title(title)
    ax.set_ylabel("Appliance energy use (Wh)")
    ax.set_xlabel("Date")
    ax.legend()

    fig.tight_layout()

    return fig


def plot_forecast_with_intervals(train, actual, forecast_df, forecast_col="sarimax",
                                  context=72, alpha_label="95% interval",
                                  title="Forecast with confidence intervals"):
    """
    Plotting a point forecast with its prediction interval against the
    realised values, showing `context` hours of training data for lead-in.
    """

    fig, ax = plt.subplots(figsize=(14, 6))

    train.tail(context).plot(ax=ax, label="Training data", color="tab:blue", linewidth=1.5)
    actual.plot(ax=ax, label="Actual", color="black", linewidth=2.0)

    forecast_df[forecast_col].plot(ax=ax, label="Forecast", color="tab:red", linewidth=2.0)

    ax.fill_between(
        forecast_df.index,
        forecast_df["lower"],
        forecast_df["upper"],
        color="tab:red",
        alpha=0.2,
        label=alpha_label,
    )

    ax.set_title(title)
    ax.set_ylabel("Appliance energy use (Wh)")
    ax.set_xlabel("Date")
    ax.legend()

    fig.tight_layout()

    return fig


def plot_residual_diagnostics(resid, lags=48, title="Residual diagnostics"):
    """
    Four-panel residual check: residuals over time, ACF, histogram
    against a fitted normal, and a normal Q-Q plot.
    """

    resid = resid.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(resid.index, resid.values, linewidth=0.8)
    axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residuals over time")
    axes[0, 0].set_ylabel("Residual")

    plot_acf(resid, lags=lags, ax=axes[0, 1])
    axes[0, 1].set_title("Residual ACF")

    axes[1, 0].hist(resid.values, bins=50, density=True, alpha=0.7,
                    color="tab:blue", edgecolor="white")
    x_grid = np.linspace(resid.min(), resid.max(), 200)
    axes[1, 0].plot(x_grid, stats.norm.pdf(x_grid, resid.mean(), resid.std()),
                    color="tab:red", linewidth=2, label="Normal fit")
    axes[1, 0].set_title("Residual distribution")
    axes[1, 0].set_xlabel("Residual")
    axes[1, 0].legend()

    stats.probplot(resid.values, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Normal Q-Q plot")

    fig.suptitle(title)
    fig.tight_layout()

    return fig


def save_fig(fig, path, dpi=300):
    """Save a figure to `path`, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {path}")
