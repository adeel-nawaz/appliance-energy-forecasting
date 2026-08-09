"""
Data loading, cleaning, and resampling for the appliance energy dataset.

Pipeline:
    raw 10-minute CSV -> parsed/cleaned -> resampled to hourly -> saved
    to data/processed/appliance_hourly.csv
"""

import urllib.request
from pathlib import Path

import pandas as pd

from appliance_energy import config


def download_raw_data(dest_path=config.RAW_DATA_PATH, url=config.RAW_DATA_URL,
                       force=False):
    """
    Download the raw Appliances Energy Prediction CSV from UCI if it is
    not already present locally.
    """

    dest_path = Path(dest_path) if not isinstance(dest_path, Path) else dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        print(f"Raw data already present at {dest_path}, skipping download.")
        return dest_path

    print(f"Downloading raw data from {url} ...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved raw data to {dest_path}")

    return dest_path


def load_raw_data(path=config.RAW_DATA_PATH):
    """
    Load the raw dataset from a local CSV, parse the timestamp, and set
    it as a sorted DatetimeIndex.
    """

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    return df


def clean_raw_data(df, target=config.TARGET):
    """
    Coerce all columns to numeric and drop rows with a missing target.

    Non-target missing values are left for resampling/interpolation to
    handle, since dropping them here would fragment the time index.
    """

    out = df.copy()

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=[target])

    return out


def resample_hourly(df):
    """
    Resample a cleaned, 10-minute-indexed dataframe to hourly means.

    Any gaps introduced by resampling are filled with time-based
    interpolation, then any remaining edge NaNs are dropped.
    """

    hourly = df.resample("h").mean()
    hourly = hourly.interpolate("time")
    hourly = hourly.dropna()

    return hourly


def prepare_hourly_data(raw_path=config.RAW_DATA_PATH,
                         save_path=config.HOURLY_DATA_PATH,
                         target=config.TARGET):
    """
    Full raw-to-hourly preparation pipeline. Loads the raw CSV, cleans
    it, resamples to hourly, and writes the processed CSV to disk.
    """

    config.ensure_dirs()

    raw = load_raw_data(raw_path)
    cleaned = clean_raw_data(raw, target=target)
    hourly = resample_hourly(cleaned)

    hourly.to_csv(save_path)
    print(f"Saved processed hourly data to {save_path} ({hourly.shape[0]} rows)")

    return hourly


def load_hourly_data(save_path=config.HOURLY_DATA_PATH,
                      raw_path=config.RAW_DATA_PATH,
                      target=config.TARGET,
                      force_rebuild=False):
    """
    Load the processed hourly dataset, building it from raw data first
    if it does not exist yet (or force_rebuild=True).
    """

    if save_path.exists() and not force_rebuild:
        hourly = pd.read_csv(save_path, index_col=0, parse_dates=True)
        return hourly

    return prepare_hourly_data(raw_path=raw_path, save_path=save_path, target=target)


def train_test_split_series(series, test_steps=config.TEST_STEPS):
    """Split a series into train/test using the final `test_steps` rows as test."""

    train = series.iloc[:-test_steps]
    test = series.iloc[-test_steps:]

    return train, test


def train_test_split_frame(df, test_steps=config.TEST_STEPS):
    """Split a dataframe into train/test using the final `test_steps` rows as test."""

    train = df.iloc[:-test_steps]
    test = df.iloc[-test_steps:]

    return train, test
