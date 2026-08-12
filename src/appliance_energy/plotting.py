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


def plot_target_correlations(table, target, top_n=20,
                              title="Correlation with the target"):
    """
    Horizontal bar chart of each feature's correlation with the target,
    showing the `top_n` strongest by absolute value.
    """

    correlations = (
        table.corr(numeric_only=True)[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
        .head(top_n)
        .sort_values()
    )

    colors = ["tab:red" if v < 0 else "tab:blue" for v in correlations]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(correlations))))
    ax.barh(correlations.index, correlations.values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Pearson correlation")

    fig.tight_layout()

    return fig, correlations


def plot_group_sizes(groups, title="Features by availability group"):
    """Bar chart of how many features fall into each availability group."""

    names = list(groups)
    counts = [len(groups[name]) for name in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, counts, color="tab:blue")
    ax.set_title(title)
    ax.set_ylabel("Number of features")

    for i, count in enumerate(counts):
        ax.text(i, count + 0.3, str(count), ha="center")

    fig.tight_layout()

    return fig


def plot_feature_importance(importances, top_n=25,
                             title="Feature importance"):
    """Horizontal bar chart of the `top_n` most important features."""

    top = importances.sort_values(ascending=False).head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(top))))
    ax.barh(top.index, top.values, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Importance")

    fig.tight_layout()

    return fig


def plot_ablation(ablation, metric="MASE", baseline=None,
                   title="Cumulative feature-group ablation"):
    """
    Plot how a metric evolves as feature groups are added.

    Bars are coloured by whether the group would genuinely be known at
    the forecast origin, so the operational cost of each gain is visible.
    """

    labels = [f"+{row}" for row in ablation["added_group"]]
    colors = [
        "tab:blue" if known else "tab:orange"
        for known in ablation["known_at_origin"]
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, ablation[metric], color=colors)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="tab:blue", label="known at forecast origin"),
        plt.Rectangle((0, 0), 1, 1, color="tab:orange", label="requires realised values"),
    ]

    if baseline is not None:
        line = ax.axhline(baseline, color="black", linestyle="--", linewidth=1.5,
                          label=f"strongest benchmark ({baseline:.3f})")
        handles.append(line)

    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel("Feature groups added (cumulative)")
    ax.legend(handles=handles, fontsize=8)

    fig.tight_layout()

    return fig


def plot_error_diagnostics(errors, hourly_errors, step_errors, summary,
                            title="Forecast error diagnostics"):
    """
    Four-panel error diagnostic across models.

    Panel 1: absolute error by hour of day -- when do models fail?
    Panel 2: absolute error by step ahead  -- how does error grow with distance?
    Panel 3: signed error distributions    -- is the model biased?
    Panel 4: error percentiles             -- typical vs worst-case behaviour.
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    hourly_errors.plot(ax=axes[0, 0], marker="o", linewidth=1.5)
    axes[0, 0].set_title("Mean absolute error by hour of day")
    axes[0, 0].set_xlabel("Hour")
    axes[0, 0].set_ylabel("MAE (Wh)")
    axes[0, 0].legend(fontsize=7)

    step_errors.plot(ax=axes[0, 1], marker="o", linewidth=1.5)
    axes[0, 1].set_title("Mean absolute error by steps ahead of origin")
    axes[0, 1].set_xlabel("Steps ahead")
    axes[0, 1].set_ylabel("MAE (Wh)")
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].boxplot(
        [errors[col].dropna() for col in errors.columns],
        labels=list(errors.columns),
        showfliers=False,
    )
    axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Signed error distribution (forecast - actual)")
    axes[1, 0].set_ylabel("Error (Wh)")
    axes[1, 0].tick_params(axis="x", rotation=45, labelsize=7)

    summary[["median", "p90", "p95"]].plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title("Absolute error percentiles")
    axes[1, 1].set_ylabel("Absolute error (Wh)")
    axes[1, 1].tick_params(axis="x", rotation=45, labelsize=7)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    return fig


def plot_metric_comparison(results, baseline_mase=None,
                            metrics=("MAE", "RMSE", "MASE"),
                            title="Model comparison"):
    """Grouped bar chart of each metric across models, sorted by MASE."""

    ordered = results.sort_values("MASE")

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        colors = ["tab:green" if m == ordered["MASE"].min() else "tab:blue"
                  for m in ordered["MASE"]]
        ax.barh(ordered["model"], ordered[metric], color=colors)
        ax.invert_yaxis()
        ax.set_title(metric)
        ax.tick_params(axis="y", labelsize=8)

        if metric == "MASE":
            ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2,
                       label="MASE = 1")
            if baseline_mase is not None:
                ax.axvline(baseline_mase, color="black", linestyle=":",
                           linewidth=1.2, label="strongest benchmark")
            ax.legend(fontsize=7)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    return fig


def plot_skill_scores(skill, metric="MASE_improvement_%",
                       title="Improvement over the strongest benchmark"):
    """Bar chart of percentage improvement over a baseline, signed."""

    ordered = skill.sort_values(metric)
    colors = ["tab:green" if v > 0 else "tab:red" for v in ordered[metric]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(ordered["model"], ordered[metric], color=colors)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("% improvement (positive = better than benchmark)")
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()

    return fig


def save_fig(fig, path, dpi=300):
    """Save a figure to `path`, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {path}")
