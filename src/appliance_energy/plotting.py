"""Shared plotting utilities for time-series EDA and forecast comparison."""

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_time_series(series, title="", ylabel="", ax=None, **kwargs):
    """Plot a single time series, optionally onto an existing axis."""

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
    """Plot a statsmodels DecomposeResult (observed/trend/seasonal/resid)."""

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


def save_fig(fig, path, dpi=300):
    """Save a figure to `path`, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {path}")
