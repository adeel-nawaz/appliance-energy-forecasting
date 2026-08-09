# scripts/download_data.py
#
# Download the raw Appliances Energy Prediction dataset (if not already
# present) and build the processed hourly dataset used by the rest of
# the pipeline.
#
# Usage:
#   python scripts/download_data.py

from appliance_energy import config, data


def main():
    config.ensure_dirs()
    data.download_raw_data()
    data.prepare_hourly_data()


if __name__ == "__main__":
    main()
