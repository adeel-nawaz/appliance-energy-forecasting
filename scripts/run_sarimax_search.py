# scripts/run_sarimax_search.py
#
# AIC-based SARIMA order selection.
#
# The assignment requires looping over every combination of
# p in [0, 6], d in [0, 2], q in [0, 6] (147 models).
# The search runs in two stages:
#
#   Stage 1: all 147 non-seasonal (p, d, q) combinations, ranked by AIC.
#   Stage 2: the top-K orders from stage 1 refitted across a small
#            seasonal grid (P, D, Q) at period 24, ranked by AIC again.
#
# Results are cached to outputs/metrics/ so the notebooks never refit.
#
# Usage:
#   python scripts/run_sarimax_search.py
#   python scripts/run_sarimax_search.py --n-jobs 4
#   python scripts/run_sarimax_search.py --skip-stage1     # reuse cached stage 1
#   python scripts/run_sarimax_search.py --top-k 2 --search-maxiter 30

import argparse
import warnings

warnings.filterwarnings("ignore")

from appliance_energy import config, data
from appliance_energy.models import sarimax


def parse_args():
    parser = argparse.ArgumentParser(description="SARIMA AIC order search.")

    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel fits (-1 uses all cores). Default 1.")
    parser.add_argument("--top-k", type=int, default=config.SEASONAL_REFINE_TOP_K,
                        help="How many verified orders to refine seasonally.")
    parser.add_argument("--verify-top-n", type=int, default=config.VERIFY_TOP_N,
                        help="How many stage-1 orders to refit at full maxiter.")
    parser.add_argument("--search-maxiter", type=int, default=config.SEARCH_MAXITER,
                        help="Optimiser iteration cap during the search.")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Reuse cached stage-1 results instead of refitting.")
    parser.add_argument("--skip-stage2", action="store_true",
                        help="Stop after stage 1 (no seasonal refinement).")

    return parser.parse_args()


def main():
    args = parse_args()
    config.ensure_dirs()

    # Train on everything before the 14-day test period so that order
    # selection never sees the test data.
    hourly = data.load_hourly_data()
    y = hourly[config.TARGET].asfreq("h")
    train, test = data.train_test_split_series(y, test_steps=config.TEST_STEPS)

    print(f"Training observations: {len(train)}")
    print(f"Train period: {train.index.min()} to {train.index.max()}\n")

    # ----------------------------------------------------------------
    # Stage 1: full non-seasonal (p, d, q) grid
    # ----------------------------------------------------------------

    if args.skip_stage1 and config.SARIMAX_STAGE1_PATH.exists():
        import pandas as pd
        stage1 = pd.read_csv(config.SARIMAX_STAGE1_PATH)
        print(f"Loaded cached stage-1 results from {config.SARIMAX_STAGE1_PATH}\n")
    else:
        print("=" * 60)
        print("STAGE 1: non-seasonal (p, d, q) grid")
        print("=" * 60)

        stage1 = sarimax.grid_search_orders(
            train,
            p_range=config.P_RANGE,
            d_range=config.D_RANGE,
            q_range=config.Q_RANGE,
            seasonal_order=None,
            maxiter=args.search_maxiter,
            n_jobs=args.n_jobs,
        )

        stage1.to_csv(config.SARIMAX_STAGE1_PATH, index=False)
        print(f"\nSaved stage-1 results to {config.SARIMAX_STAGE1_PATH}")

    n_converged = int(stage1["converged"].sum())
    print(f"\nConverged: {n_converged}/{len(stage1)}")
    print("\nTop 10 non-seasonal orders by AIC (screening pass):")
    print(stage1.head(10)[["order", "aic", "bic", "converged", "fit_seconds"]].to_string(index=False))

    # ----------------------------------------------------------------
    # Verification: refit the leading candidates with a higher maxiter.
    #
    # The screening pass uses a low iteration cap for speed, which leaves
    # the more complex models unconverged and their AIC unreliable.
    # Selection must be made on converged likelihoods.
    # ----------------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"VERIFICATION: refit top {args.verify_top_n} with maxiter={config.FINAL_MAXITER}")
    print("=" * 60)

    verified = sarimax.verify_top_orders(
        train, stage1, top_n=args.verify_top_n,
        seasonal_order=None, maxiter=config.FINAL_MAXITER, n_jobs=args.n_jobs,
    )

    verified.to_csv(config.SARIMAX_STAGE1_VERIFIED_PATH, index=False)
    print(f"\nSaved verified results to {config.SARIMAX_STAGE1_VERIFIED_PATH}")

    print("\nVerified ranking:")
    print(verified[["order", "aic", "bic", "converged", "fit_seconds"]].to_string(index=False))

    if args.skip_stage2:
        print("\nStopping after stage 1 (--skip-stage2).")
        return

    # ----------------------------------------------------------------
    # Stage 2: seasonal refinement of the best verified orders
    # ----------------------------------------------------------------

    converged_only = verified[verified["converged"].astype(bool)]
    pool = converged_only if not converged_only.empty else verified

    valid = pool.head(args.top_k)

    candidate_orders = [
        (int(row["p"]), int(row["d"]), int(row["q"]))
        for _, row in valid.iterrows()
    ]

    # Stage 1 had no seasonal term to work with, so it inflates the AR/MA
    # order to imitate the daily cycle. Add parsimonious candidates so the
    # seasonal search can find a simpler model if one fits just as well.
    for extra in config.SEASONAL_EXTRA_ORDERS:
        if extra not in candidate_orders:
            candidate_orders.append(extra)

    print("\n" + "=" * 60)
    print("STAGE 2: seasonal refinement at period 24")
    print("=" * 60)
    print(f"Candidate orders carried forward: {candidate_orders}")

    stage2 = sarimax.refine_seasonal_orders(
        train,
        candidate_orders=candidate_orders,
        seasonal_p_range=config.SEASONAL_P_RANGE,
        seasonal_d_range=config.SEASONAL_D_RANGE,
        seasonal_q_range=config.SEASONAL_Q_RANGE,
        period=config.DAILY_PERIOD,
        maxiter=args.search_maxiter,
        n_jobs=args.n_jobs,
        simple_differencing=True,
    )

    stage2.to_csv(config.SARIMAX_STAGE2_PATH, index=False)
    print(f"\nSaved stage-2 results to {config.SARIMAX_STAGE2_PATH}")

    print("\nAll seasonal candidates by AIC:")
    print(stage2[["order", "seasonal_order", "aic", "bic", "converged", "fit_seconds"]].to_string(index=False))

    # ----------------------------------------------------------------
    # Refit the winner properly and cache it
    # ----------------------------------------------------------------

    order, seasonal_order = sarimax.best_order_from_results(stage2)

    print("\n" + "=" * 60)
    print(f"BEST MODEL: SARIMA{order}x{seasonal_order}")
    print("=" * 60)
    # The search used simple_differencing for speed; the model we keep is
    # refitted in full state-space form (simple_differencing=False) so
    # that it uses every observation and forecasts on the original scale.
    print(f"Refitting in full state-space form with maxiter={config.FINAL_MAXITER} ...")

    best_fit = sarimax.fit_sarimax(
        train, order=order, seasonal_order=seasonal_order,
        maxiter=config.FINAL_MAXITER, simple_differencing=False,
    )

    print(f"\nAIC={best_fit.aic:.2f}  BIC={best_fit.bic:.2f}")
    print(best_fit.summary())

    sarimax.save_model(best_fit)
    sarimax.save_best_params(
        order, seasonal_order,
        extra={
            "aic": float(best_fit.aic),
            "bic": float(best_fit.bic),
            "n_train": int(len(train)),
            "selection": "AIC, two-stage grid search",
        },
    )


if __name__ == "__main__":
    main()
