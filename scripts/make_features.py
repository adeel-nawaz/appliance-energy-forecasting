# scripts/make_features.py
#
# Build the supervised-learning feature table from the processed hourly
# data and save it to data/processed/feature_table.csv.
#
# All lag and rolling features respect the 24-hour forecast horizon, so
# no feature uses a target value that would be unobserved at the
# forecast origin.
#
# Usage:
#   python scripts/make_features.py
#   python scripts/make_features.py --horizon 24
#   python scripts/make_features.py --include-random   # keep rv1/rv2

import argparse
import warnings

warnings.filterwarnings("ignore")

from appliance_energy import config, data, features


def parse_args():
    parser = argparse.ArgumentParser(description="Build the feature table.")

    parser.add_argument("--horizon", type=int, default=config.HORIZON,
                        help="Forecast horizon in hours (controls lag safety).")
    parser.add_argument("--include-random", action="store_true",
                        help="Keep the dataset's synthetic rv1/rv2 columns.")
    parser.add_argument("--allow-leaky", action="store_true",
                        help="Keep sub-horizon lags. For leakage demonstration only.")

    return parser.parse_args()


def main():
    args = parse_args()
    config.ensure_dirs()

    hourly = data.load_hourly_data()

    table = features.build_feature_table(
        hourly,
        target=config.TARGET,
        horizon=args.horizon,
        allow_leaky=args.allow_leaky,
        include_random=args.include_random,
    )

    groups = features.feature_groups(table.columns)

    print(f"Feature table: {table.shape[0]} rows x {table.shape[1]} columns")
    print(f"Rows dropped to warm up lags/rolling windows: {len(hourly) - len(table)}\n")

    for name, cols in groups.items():
        print(f"  {name:9s} ({len(cols):2d}): {', '.join(cols)}")

    known = features.known_at_origin_columns(table.columns)
    print(f"\nKnown at forecast origin : {len(known)} features")
    print(f"Requires realised values : "
          f"{len([c for c in table.columns if c != config.TARGET]) - len(known)} features "
          f"(using these gives a conditional forecast)")

    table.to_csv(config.FEATURE_TABLE_PATH)
    print(f"\nSaved feature table to {config.FEATURE_TABLE_PATH}")


if __name__ == "__main__":
    main()
