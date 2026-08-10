"""
Rolling-origin (walk-forward) backtesting.

Why this module exists
----------------------
The forecasting task is "forecast the next 24 hours", but the test
period is the final 14 days (336 hours). Producing a single 336-step
forecast would measure 336-step-ahead accuracy, which is a much harder
and quite different problem from the one the assignment poses.

Rolling-origin evaluation resolves this: forecast `horizon` steps from
the current origin, then reveal the realised values for that block,
advance the origin, and repeat. Every prediction is therefore at most
`horizon` steps ahead, while the evaluation still spans the whole test
period. This is the standard way to evaluate a fixed-horizon forecast
over an extended test window, and it keeps every model -- benchmarks,
SARIMAX, feature-based, foundation -- on an identical footing.

No leakage is introduced: at each origin the model only ever sees data
strictly before the block it is predicting.
"""

import numpy as np
import pandas as pd

from appliance_energy import config


def rolling_origin_forecast(forecast_fn, y_train, y_test,
                            horizon=config.HORIZON, name=None, **kwargs):
    """
    Walk-forward forecasting for any function with the benchmark
    signature ``forecast_fn(y_train, horizon, index, **kwargs)``.

    At each origin the function is given every observation up to that
    point (training data plus the test blocks already revealed) and asked
    for the next `horizon` steps.

    Returns a single series covering the whole of `y_test`.
    """

    predictions = []
    history = y_train.copy()

    n_blocks = int(np.ceil(len(y_test) / horizon))

    for block_i in range(n_blocks):
        start = block_i * horizon
        block = y_test.iloc[start:start + horizon]

        prediction = forecast_fn(history, len(block), block.index, **kwargs)
        predictions.append(pd.Series(np.asarray(prediction), index=block.index))

        # Reveal the realised values before moving to the next origin.
        history = pd.concat([history, block])

    out = pd.concat(predictions)

    if name:
        out.name = name

    return out


def rolling_origin_indices(y_test, horizon=config.HORIZON):
    """
    Yield the (start, end) positional slices of each forecast block.

    Useful for reporting per-block metrics or for models that need to
    rebuild features at each origin.
    """

    n_blocks = int(np.ceil(len(y_test) / horizon))

    return [
        (block_i * horizon, min((block_i + 1) * horizon, len(y_test)))
        for block_i in range(n_blocks)
    ]
