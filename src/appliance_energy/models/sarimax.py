"""
SARIMA / SARIMAX modelling: fitting, AIC-based order search, forecasting
with confidence intervals, and residual diagnostics.

Notes on the order search
-------------------------
AIC is only strictly comparable across models fitted to the *same*
observations. Differencing changes the effective sample size, so
comparing AIC across different `d` is not automatically valid. We keep
`simple_differencing=False` (the statsmodels default) so that
differencing is handled inside the state space and the likelihood is
evaluated over a consistent sample, which keeps AIC comparable across
the `d` grid the assignment asks us to search.
"""

import json
import pickle
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from appliance_energy import config


def _trend_for_order(order, seasonal_order=None):
    """
    Choose the deterministic trend term.

    A constant is only meaningful when the series is not differenced;
    with d > 0 or D > 0 a constant becomes a drift and is dropped.
    """

    d = order[1]
    seasonal_d = seasonal_order[1] if seasonal_order else 0

    return "c" if (d == 0 and seasonal_d == 0) else "n"


def fit_sarimax(y_train, exog=None, order=config.SARIMAX_ORDER,
                seasonal_order=config.SARIMAX_SEASONAL_ORDER,
                trend=None, maxiter=config.FINAL_MAXITER,
                enforce_stationarity=False, enforce_invertibility=False,
                simple_differencing=False):
    """
    Fit a single SARIMAX model.

    `seasonal_order=None` (or a zero seasonal order) fits a plain,
    non-seasonal ARIMA. `exog` adds exogenous regressors, turning the
    SARIMA into a SARIMAX.
    """

    if seasonal_order is None:
        seasonal_order = (0, 0, 0, 0)

    if trend is None:
        trend = _trend_for_order(order, seasonal_order)

    model = SARIMAX(
        y_train,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        simple_differencing=simple_differencing,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(disp=False, maxiter=maxiter)

    return fit


def _evaluate_order(y_train, order, seasonal_order, exog=None,
                    maxiter=config.SEARCH_MAXITER, simple_differencing=False):
    """
    Fit one candidate order and return its information criteria.

    Failures are captured rather than raised so that a single badly
    conditioned combination cannot abort the whole grid search.

    `simple_differencing=True` differences the data up front instead of
    inside the state space. That is dramatically faster for seasonal
    models (it removes s*D states) but discards the first s*D + d
    observations, so AIC is then only comparable between models sharing
    the same d and D. `nobs` is recorded on every row so this can be
    checked rather than assumed.
    """

    start = time.time()

    record = {
        "p": order[0], "d": order[1], "q": order[2],
        "P": seasonal_order[0] if seasonal_order else 0,
        "D": seasonal_order[1] if seasonal_order else 0,
        "Q": seasonal_order[2] if seasonal_order else 0,
        "s": seasonal_order[3] if seasonal_order else 0,
        "order": str(order),
        "seasonal_order": str(seasonal_order) if seasonal_order else "(0, 0, 0, 0)",
    }

    try:
        fit = fit_sarimax(
            y_train, exog=exog, order=order, seasonal_order=seasonal_order,
            maxiter=maxiter, simple_differencing=simple_differencing,
        )

        record.update({
            "aic": float(fit.aic),
            "bic": float(fit.bic),
            "hqic": float(fit.hqic),
            "llf": float(fit.llf),
            "nobs": int(fit.nobs),
            "converged": bool(fit.mle_retvals.get("converged", False)),
            "error": "",
        })

    except Exception as exc:  # noqa: BLE001 - want the search to continue
        record.update({
            "aic": np.nan, "bic": np.nan, "hqic": np.nan, "llf": np.nan,
            "nobs": np.nan, "converged": False,
            "error": f"{type(exc).__name__}: {exc}",
        })

    record["fit_seconds"] = round(time.time() - start, 2)

    return record


def grid_search_orders(y_train, p_range=config.P_RANGE, d_range=config.D_RANGE,
                       q_range=config.Q_RANGE, seasonal_order=None, exog=None,
                       maxiter=config.SEARCH_MAXITER, n_jobs=1, verbose=True):
    """
    Exhaustively search (p, d, q) by AIC.

    Covers the full grid the assignment specifies. `seasonal_order=None`
    searches non-seasonal ARIMA models (stage 1); passing a fixed
    seasonal order searches (p, d, q) alongside it.

    Returns a dataframe of every combination sorted by AIC (ascending).
    """

    combinations = list(product(p_range, d_range, q_range))

    if verbose:
        print(f"Searching {len(combinations)} (p, d, q) combinations "
              f"(seasonal_order={seasonal_order}) ...")

    def run(order):
        return _evaluate_order(y_train, order, seasonal_order, exog=exog, maxiter=maxiter)

    if n_jobs == 1:
        records = []
        for i, order in enumerate(combinations, start=1):
            record = run(order)
            records.append(record)
            if verbose:
                aic = record["aic"]
                aic_txt = f"{aic:10.2f}" if np.isfinite(aic) else "      FAIL"
                print(f"  [{i:3d}/{len(combinations)}] ARIMA{order}  "
                      f"AIC={aic_txt}  ({record['fit_seconds']:.1f}s)", flush=True)
    else:
        from joblib import Parallel, delayed

        records = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(run)(order) for order in combinations
        )

    results = pd.DataFrame(records)

    return results.sort_values("aic", na_position="last").reset_index(drop=True)


def refine_seasonal_orders(y_train, candidate_orders,
                           seasonal_p_range=config.SEASONAL_P_RANGE,
                           seasonal_d_range=config.SEASONAL_D_RANGE,
                           seasonal_q_range=config.SEASONAL_Q_RANGE,
                           period=config.DAILY_PERIOD, exog=None,
                           maxiter=config.SEARCH_MAXITER, n_jobs=1, verbose=True,
                           simple_differencing=True):
    """
    Stage 2 of the search: take the best non-seasonal orders and try a
    small seasonal grid at the given period, ranking again by AIC.

    `simple_differencing` defaults to True here purely for speed: a
    seasonal fit at period 24 drops from minutes to seconds. This is
    safe as long as every candidate shares the same d and D, which the
    caller should ensure.
    """

    seasonal_orders = [
        (P, D, Q, period)
        for P, D, Q in product(seasonal_p_range, seasonal_d_range, seasonal_q_range)
    ]

    # With simple_differencing the effective sample depends on d and D,
    # so AIC is only comparable when those are constant across the grid.
    if simple_differencing:
        distinct_d = {order[1] for order in candidate_orders}
        distinct_seasonal_d = {seasonal[1] for seasonal in seasonal_orders}

        if len(distinct_d) > 1 or len(distinct_seasonal_d) > 1:
            warnings.warn(
                "simple_differencing=True with varying d "
                f"({sorted(distinct_d)}) or D ({sorted(distinct_seasonal_d)}) "
                "means models are fitted to different sample sizes, so their "
                "AIC values are not comparable. Pass simple_differencing=False "
                "for a valid comparison.",
                stacklevel=2,
            )

    jobs = list(product(candidate_orders, seasonal_orders))

    if verbose:
        print(f"Refining {len(candidate_orders)} order(s) against "
              f"{len(seasonal_orders)} seasonal order(s) = {len(jobs)} fits ...")

    def run(job):
        order, seasonal_order = job
        return _evaluate_order(y_train, order, seasonal_order, exog=exog,
                               maxiter=maxiter, simple_differencing=simple_differencing)

    if n_jobs == 1:
        records = []
        for i, job in enumerate(jobs, start=1):
            record = run(job)
            records.append(record)
            if verbose:
                aic = record["aic"]
                aic_txt = f"{aic:10.2f}" if np.isfinite(aic) else "      FAIL"
                print(f"  [{i:3d}/{len(jobs)}] SARIMA{job[0]}x{job[1]}  "
                      f"AIC={aic_txt}  ({record['fit_seconds']:.1f}s)", flush=True)
    else:
        from joblib import Parallel, delayed

        records = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(run)(job) for job in jobs
        )

    results = pd.DataFrame(records)

    return results.sort_values("aic", na_position="last").reset_index(drop=True)


def verify_top_orders(y_train, results, top_n=config.VERIFY_TOP_N,
                      seasonal_order=None, exog=None,
                      maxiter=config.FINAL_MAXITER, n_jobs=1, verbose=True):
    """
    Re-fit the most promising orders with a higher iteration cap.

    The screening grid uses a low `maxiter` so that 147 fits stay
    affordable, but that leaves the more heavily parameterised models
    short of convergence, and an AIC taken from a stopped optimisation
    is not trustworthy. This refits the leading candidates properly and
    re-ranks them, so selection is made on converged likelihoods.
    """

    candidates = results.dropna(subset=["aic"]).head(top_n)

    orders = [
        (int(row["p"]), int(row["d"]), int(row["q"]))
        for _, row in candidates.iterrows()
    ]

    if verbose:
        print(f"Re-fitting the top {len(orders)} orders with maxiter={maxiter} ...")

    def run(order):
        return _evaluate_order(y_train, order, seasonal_order, exog=exog, maxiter=maxiter)

    if n_jobs == 1:
        records = []
        for i, order in enumerate(orders, start=1):
            record = run(order)
            records.append(record)
            if verbose:
                status = "converged" if record["converged"] else "NOT converged"
                aic = record["aic"]
                aic_txt = f"{aic:10.2f}" if np.isfinite(aic) else "      FAIL"
                print(f"  [{i:2d}/{len(orders)}] ARIMA{order}  AIC={aic_txt}  "
                      f"{status}  ({record['fit_seconds']:.1f}s)", flush=True)
    else:
        from joblib import Parallel, delayed

        records = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(run)(order) for order in orders
        )

    verified = pd.DataFrame(records)

    return verified.sort_values("aic", na_position="last").reset_index(drop=True)


def best_order_from_results(results, require_converged=True):
    """
    Extract ((p,d,q), (P,D,Q,s)) from the best row of a search results frame.
    """

    valid = results.dropna(subset=["aic"])

    if require_converged and "converged" in valid.columns:
        converged = valid[valid["converged"].astype(bool)]

        if converged.empty:
            print("Warning: no model converged; falling back to AIC ranking "
                  "over non-converged fits.")
        else:
            valid = converged

    if valid.empty:
        raise ValueError("No SARIMAX model in the search results converged.")

    valid = valid.sort_values("aic")
    best = valid.iloc[0]

    order = (int(best["p"]), int(best["d"]), int(best["q"]))
    seasonal_order = (int(best["P"]), int(best["D"]), int(best["Q"]), int(best["s"]))

    return order, seasonal_order


# ------------------------------------------------------------------
# Forecasting
# ------------------------------------------------------------------

def forecast_sarimax(fit, horizon, index, exog=None, alpha=0.05, name="sarimax"):
    """
    Forecast `horizon` steps ahead with prediction intervals.

    Returns a dataframe with the point forecast and the lower/upper
    bounds of the (1 - alpha) interval, indexed by `index`.
    """

    forecast = fit.get_forecast(steps=horizon, exog=exog)

    mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=alpha)

    out = pd.DataFrame({
        name: np.asarray(mean),
        "lower": np.asarray(conf_int.iloc[:, 0]),
        "upper": np.asarray(conf_int.iloc[:, 1]),
    }, index=index)

    return out


def rolling_forecast_sarimax(fit, y_test, horizon=config.HORIZON, exog_test=None,
                             alpha=0.05, refit=False, name="sarimax", verbose=False):
    """
    Walk-forward (rolling-origin) forecasting across the test period.

    we forecast `horizon` steps, append the realised observations, and
    repeat -- so every prediction is genuinely at most `horizon` steps
    ahead, while still covering the whole test period.

    `refit=False` extends the state with new observations but keeps the
    estimated parameters fixed, which is far cheaper than re-estimating
    and avoids re-selecting the model on test data.
    """

    means, lowers, uppers = [], [], []

    current = fit
    n_blocks = int(np.ceil(len(y_test) / horizon))

    for block_i in range(n_blocks):
        start = block_i * horizon
        block = y_test.iloc[start:start + horizon]

        exog_block = None
        if exog_test is not None:
            exog_block = exog_test.iloc[start:start + horizon]

        forecast = current.get_forecast(steps=len(block), exog=exog_block)
        conf_int = forecast.conf_int(alpha=alpha)

        means.append(pd.Series(np.asarray(forecast.predicted_mean), index=block.index))
        lowers.append(pd.Series(np.asarray(conf_int.iloc[:, 0]), index=block.index))
        uppers.append(pd.Series(np.asarray(conf_int.iloc[:, 1]), index=block.index))

        if verbose:
            print(f"  block {block_i + 1}/{n_blocks}: {block.index[0]} "
                  f"({len(block)} steps)", flush=True)

        # Feed the realised values in before forecasting the next block.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            current = current.append(block, exog=exog_block, refit=refit)

    return pd.DataFrame({
        name: pd.concat(means),
        "lower": pd.concat(lowers),
        "upper": pd.concat(uppers),
    })


# ------------------------------------------------------------------
# Residual diagnostics
# ------------------------------------------------------------------

def residual_diagnostics(fit, lags=48, alpha=0.05):
    """
    Summarise model residuals.

    Ljung-Box tests whether residual autocorrelation remains (a p-value
    above alpha means we cannot reject "residuals are uncorrelated",
    i.e. the model has captured the structure). Jarque-Bera tests
    normality of the residual distribution.
    """

    resid = pd.Series(fit.resid).dropna()

    # Drop the burn-in caused by differencing, which is not informative.
    resid = resid.iloc[fit.loglikelihood_burn:] if fit.loglikelihood_burn else resid

    ljung = acorr_ljungbox(resid, lags=[min(lags, len(resid) // 2)], return_df=True)
    lb_stat = float(ljung["lb_stat"].iloc[0])
    lb_pvalue = float(ljung["lb_pvalue"].iloc[0])

    jb_stat, jb_pvalue, skew, kurtosis = fit.test_normality(method="jarquebera")[0]

    return {
        "n_resid": int(len(resid)),
        "mean": float(resid.mean()),
        "std": float(resid.std()),
        "skew": float(skew),
        "kurtosis": float(kurtosis),
        "ljung_box_stat": lb_stat,
        "ljung_box_pvalue": lb_pvalue,
        "residuals_uncorrelated": bool(lb_pvalue > alpha),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "residuals_normal": bool(jb_pvalue > alpha),
    }


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def save_model(fit, path=config.SARIMAX_MODEL_PATH):
    """Pickle a fitted results object so notebooks can reload it without refitting."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as handle:
        pickle.dump(fit, handle)

    print(f"Saved fitted SARIMAX model to {path}")

    return path


def load_model(path=config.SARIMAX_MODEL_PATH):
    """Load a pickled fitted SARIMAX results object."""

    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_best_params(order, seasonal_order, extra=None,
                     path=config.SARIMAX_BEST_PARAMS_PATH):
    """Persist the selected orders as JSON so every notebook agrees on them."""

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"order": list(order), "seasonal_order": list(seasonal_order)}

    if extra:
        payload.update(extra)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved best SARIMAX parameters to {path}")

    return payload


def load_best_params(path=config.SARIMAX_BEST_PARAMS_PATH):
    """
    Load the cached best orders, falling back to the config defaults if
    the search has not been run yet.
    """

    if not path.exists():
        print(f"No cached search results at {path}; using config defaults.")
        return config.SARIMAX_ORDER, config.SARIMAX_SEASONAL_ORDER

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    return tuple(payload["order"]), tuple(payload["seasonal_order"])
