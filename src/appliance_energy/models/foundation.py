"""
Time-series foundation model: Amazon Chronos.

How the model is used
---------------------
Chronos is applied **zero-shot**. The pretrained weights are downloaded
once and used as-is: no fitting, fine-tuning, or adaptation is performed
on the appliance data. The only thing the model ever sees of this
dataset is the context window handed to it at prediction time.

Covariates
----------
Chronos is a **univariate** model. It consumes a single history of the
target series and nothing else -- no calendar features, no sensor
readings, no weather. It therefore has strictly less information
available to it than the SARIMAX and feature-based models, which is
important when interpreting the comparison: any advantage it shows comes
from pretraining rather than from covariates.

Why Chronos rather than TimeGPT or TimesFM
------------------------------------------
TimeGPT requires an API key and network calls per forecast. Chronos runs
locally from open weights, so the pipeline stays reproducible from a
fresh clone without credentials.

Forecasts are probabilistic: the model emits sample paths, from which we
take the median as the point forecast and quantiles as the prediction
interval, mirroring the SARIMAX confidence intervals.
"""

import numpy as np
import pandas as pd

from appliance_energy import config

# Chronos-Bolt is the faster, more accurate second generation. The
# "small" variant is a reasonable accuracy/runtime trade-off on CPU.
DEFAULT_MODEL = "amazon/chronos-bolt-small"

# Chronos-Bolt supports a 2048-step context; anything longer is ignored,
# so there is no benefit to passing more than this.
MAX_CONTEXT = 2048

# Chronos-Bolt was trained on the quantile levels 0.1 ... 0.9 only.
# Asking for anything outside that range is silently clipped back to the
# nearest trained level, so a request for 0.05/0.95 would quietly return
# a 0.1/0.9 band mislabelled as 95%. We therefore use the widest interval
# the model genuinely supports: 10th to 90th percentile, an 80% interval.
#
# This is a real limitation compared with SARIMAX, whose Gaussian state
# space yields an analytic interval at any confidence level.
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
INTERVAL_LABEL = "80% interval"


class ChronosUnavailable(RuntimeError):
    """Raised when the Chronos package or its weights cannot be loaded."""


def load_chronos(model_name=DEFAULT_MODEL, device="cpu"):
    """
    Load a pretrained Chronos pipeline.

    Raises `ChronosUnavailable` with an actionable message rather than a
    bare ImportError, so the pipeline can skip the foundation model
    cleanly on a machine without the dependency.
    """

    try:
        from chronos import BaseChronosPipeline
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ChronosUnavailable(
            "chronos-forecasting is not installed. "
            "Install it with: pip install chronos-forecasting"
        ) from exc

    try:
        import torch

        pipeline = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )
    except Exception as exc:  # pragma: no cover - network/weights dependent
        raise ChronosUnavailable(
            f"Could not load Chronos weights '{model_name}': {exc}"
        ) from exc

    return pipeline


def forecast_chronos(pipeline, context, horizon, index,
                     quantile_levels=DEFAULT_QUANTILES, name="foundation_model",
                     max_context=MAX_CONTEXT):
    """
    Produce a single `horizon`-step probabilistic forecast.

    `context` is the observed history up to the forecast origin. Only
    the most recent `max_context` observations are passed, since the
    model ignores anything beyond its context window.

    Returns a dataframe with the median point forecast plus lower/upper
    quantile bounds, matching the shape returned by the SARIMAX helpers.
    """

    import torch

    history = np.asarray(context, dtype=np.float32)[-max_context:]
    tensor = torch.tensor(history).unsqueeze(0)

    quantiles, _ = pipeline.predict_quantiles(
        tensor,
        prediction_length=horizon,
        quantile_levels=list(quantile_levels),
    )

    values = quantiles[0].numpy()  # shape: (horizon, n_quantiles)

    lower_i, median_i, upper_i = 0, len(quantile_levels) // 2, len(quantile_levels) - 1

    return pd.DataFrame(
        {
            name: values[:, median_i],
            "lower": values[:, lower_i],
            "upper": values[:, upper_i],
        },
        index=index,
    )


def rolling_forecast_chronos(pipeline, y_train, y_test, horizon=config.HORIZON,
                             quantile_levels=DEFAULT_QUANTILES,
                             name="foundation_model", max_context=MAX_CONTEXT,
                             verbose=False):
    """
    Rolling-origin forecasting across the test period.

    At each origin the model receives every observation up to that point
    and predicts the next `horizon` steps; the realised values are then
    revealed before moving on. This matches the scheme used for the
    benchmarks, SARIMAX, and the feature-based model, so all four are
    evaluated on identical 24-hour-ahead terms.

    No leakage: the context passed at each origin contains only data
    strictly before the block being predicted.
    """

    frames = []
    history = pd.Series(y_train).copy()

    n_blocks = int(np.ceil(len(y_test) / horizon))

    for block_i in range(n_blocks):
        start = block_i * horizon
        block = y_test.iloc[start:start + horizon]

        frames.append(
            forecast_chronos(
                pipeline,
                context=history.values,
                horizon=len(block),
                index=block.index,
                quantile_levels=quantile_levels,
                name=name,
                max_context=max_context,
            )
        )

        if verbose:
            print(f"  block {block_i + 1}/{n_blocks}: {block.index[0]}", flush=True)

        history = pd.concat([history, block])

    return pd.concat(frames)


def is_available(model_name=DEFAULT_MODEL):
    """Check whether Chronos can be loaded, without raising."""

    try:
        load_chronos(model_name)
        return True
    except ChronosUnavailable:
        return False
