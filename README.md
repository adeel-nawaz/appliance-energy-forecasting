# Appliance Energy Forecasting

A reproducible pipeline for forecasting household appliance energy use 24 hours ahead, comparing
benchmark forecasts, SARIMAX, a feature-based regression model, and a pretrained time-series
foundation model on identical test data.

Data: the **Appliances Energy Prediction** dataset, downloaded directly from the link: https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv
— 19,735 observations at 10-minute resolution from a low-energy house in Belgium, January to May
2016, with indoor sensors, outdoor weather and a lighting meter alongside the target.

## Headline results

Final 14 days held out, 24-hour horizon, rolling origin. Errors in Wh.

| Model | MAE | RMSE | MASE | Bias | vs. best benchmark |
|---|---|---|---|---|---|
| Foundation (Chronos-Bolt-small, zero-shot) | **34.27** | 68.09 | **0.641** | −16.57 | **+21.2%** |
| Feature-based (random forest, 13 features) | 35.38 | **62.03** | 0.662 | −2.37 | +18.6% |
| SARIMAX(2,0,2)(0,1,1,24) | 36.26 | 64.74 | 0.679 | −6.33 | +16.6% |
| Weekly seasonal naive *(strongest benchmark)* | 43.46 | 81.41 | 0.813 | −13.16 | — |
| Daily seasonal naive | 48.31 | 85.57 | 0.904 | +1.75 | −11.2% |
| Mean | 50.26 | 74.94 | 0.941 | −3.29 | −15.7% |
| Naive | 85.55 | 110.39 | 1.601 | +50.98 | −96.9% |
| Drift | 85.80 | 110.68 | 1.606 | +51.37 | −97.4% |

All three advanced models beat the strongest benchmark, but by 17–21% and within three points of
each other. Two results worth flagging: the sensor and weather covariates made the feature model
**worse** (validation MASE 0.664 → 1.463), and exogenous weather did not help SARIMAX either.

## Installation

Python 3.9+ (developed on 3.10).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only, avoids CUDA build
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `appliance_energy` package so notebooks, scripts and tests import it
without path manipulation. The foundation model is optional: if `torch` or `chronos-forecasting` is
missing, the pipeline skips it and continues.

## Running the pipeline

```bash
python scripts/run_pipeline.py
```

The single entry point. It loads the hourly data (downloading the raw CSV if absent), splits off the
final 14 days, fits and forecasts the benchmarks, SARIMAX, the feature model and the foundation
model, evaluates them on identical timestamps, and saves forecasts, metrics and figures.

| Script | Purpose | Runtime |
|---|---|---|
| `download_data.py` | Fetch raw CSV, build hourly dataset | ~1 min |
| `make_features.py` | Rebuild feature table, report group availability | seconds |
| `run_pipeline.py` | Full pipeline, all models, all outputs | ~10 min* |
| `evaluate_models.py` | Re-score saved forecasts and diagnostics, no refitting | seconds |
| `run_sarimax_search.py` | AIC order search over the required grid | **~30 min** |

\* on a first run, which fits SARIMAX and downloads ~50 MB of Chronos weights. Both are cached
afterwards, so later runs take about two minutes.

The pipeline uses the selected SARIMAX order, which is recorded in
`outputs/metrics/sarimax_best_params.json`; it does not repeat the order search. To reproduce that
search: `python scripts/run_sarimax_search.py` (add `--skip-stage1` to reuse stage 1, `--n-jobs 4`
for parallel fits).

## Repository structure

```text
appliance-energy-forecasting/
│
├── README.md
├── requirements.txt
├── pyproject.toml                  # editable install of src/appliance_energy
├── .gitignore
├── Assignment report.pdf           # final report
│
├── data/
│   ├── raw/                        # raw CSV, downloaded on first run
│   ├── interim/
│   └── processed/                  # hourly dataset and feature table, built by the pipeline
│
├── notebooks/
│   ├── 01_data_download_and_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_benchmark_models.ipynb
│   ├── 04_sarimax_models.ipynb
│   ├── 05_feature_based_models.ipynb
│   ├── 06_foundation_model.ipynb
│   └── 07_model_comparison.ipynb
│
├── src/
│   └── appliance_energy/
│       ├── __init__.py
│       ├── config.py               # all paths and modelling constants
│       ├── data.py                 # download, clean, resample, split
│       ├── features.py             # calendar, lag, rolling, sensor features
│       ├── backtest.py             # rolling-origin evaluation
│       ├── eda.py                  # ADF, decomposition, differencing
│       ├── evaluation.py           # MAE/RMSE/MASE/Bias, skill scores, diagnostics
│       ├── plotting.py             # every figure in the report
│       ├── pipeline.py             # orchestration
│       └── models/
│           ├── __init__.py
│           ├── benchmarks.py       # mean, naive, seasonal naive, drift
│           ├── sarimax.py          # fit, AIC grid search, intervals, diagnostics
│           ├── feature_models.py   # XGBoost / HistGB / random forest, ablation
│           └── foundation.py       # Chronos, zero-shot
│
├── scripts/
│   ├── download_data.py
│   ├── make_features.py
│   ├── run_pipeline.py
│   ├── evaluate_models.py
│   └── run_sarimax_search.py
│
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   └── metrics/
│
└── tests/                          # 97 tests
    ├── test_backtest.py
    ├── test_benchmarks.py
    ├── test_diagnostics.py
    ├── test_evaluation.py
    ├── test_feature_models.py
    ├── test_features.py
    ├── test_foundation.py
    └── test_sarimax.py
```

## Outputs

`outputs/forecasts/all_forecasts.csv` — realised values and every model forecast, one row per test
hour: `actual, mean, naive, seasonal_naive_daily, seasonal_naive_weekly, drift, sarimax,
feature_model, foundation_model`.

`outputs/metrics/model_comparison.csv` — `model, MAE, RMSE, MASE, Bias`. Also written there:
`skill_vs_benchmark.csv`, `error_percentiles.csv`, `error_by_hour.csv`, `error_by_step_ahead.csv`,
`error_autocorrelation.csv`, `feature_group_ablation.csv`, `feature_importance.csv`, and the SARIMAX
search results.

| Figure | Shows |
|---|---|
| `forecast_comparison.png` | Every model against realised demand |
| `error_diagnostics.png` | Error by hour, by step, signed distributions, percentiles |
| `residual_acf.png` | SARIMAX residual ACF, distribution, Q–Q plot |
| `feature_importance.png` | Random-forest importances |
| `skill_scores.png` | Improvement over the strongest benchmark |

## Evaluation design

**Target** `Appliances` (hourly mean, Wh) · **Horizon** 24 hours · **Test** final 14 days (336 hours)
· **Metrics** MAE, RMSE, MASE, Bias.

The horizon is 24 hours but the test period is 336, so forecasting it in one pass would measure
336-step accuracy — a different, harder problem. Every model is instead evaluated with **rolling
origins**: forecast 24 hours, reveal the actuals, advance, repeat for 14 blocks. This matters —
evaluated over 336 steps the naive method scores MASE 4.69, but at the stated horizon it scores 1.60.

MASE is scaled by the in-sample daily seasonal-naive error, so below 1 beats that reference. Every
advanced model is compared against the **strongest benchmark**, not just against the other advanced
models.

## How leakage is prevented

| Trap | How it is handled |
|---|---|
| Future target values in lag features | Only lags ≥ horizon are admitted; `add_lag_features` refuses shorter ones and reports what it dropped |
| Rolling features without shifting first | The target is shifted by the **full horizon** before the window, not by one step |
| Scaling before the train/test split | No scaler is fitted anywhere; the tree models need none |
| Future sensor/weather values | `features.known_at_origin_columns()` marks what would genuinely be available — 19 of 50. Models using the rest are reported as **conditional forecasts** |
| Selecting on test-set performance | Feature groups and model family are chosen on a **validation block** before the test period; the test set is scored once |

The horizon rule is the easiest to get wrong. At origin `o` we predict `o+1 … o+24`, so a `lag_1`
feature on the row at `o+24` would need the target at `o+23` — inside the block being predicted.
`lag_1` correlates with the target at 0.58 against 0.31 for `lag_24`, so the leaky version scores
well and cannot be deployed.

## Notebooks

| Notebook | Covers |
|---|---|
| `01_data_download_and_cleaning` | Download, timestamp parsing, missing values, hourly resampling |
| `02_exploratory_analysis` | Daily/weekly patterns, decomposition, ACF/PACF, ADF tests |
| `03_benchmark_models` | Five benchmarks, and why the horizon changes the ranking |
| `04_sarimax_models` | Stationarity, AIC grid search, residual diagnostics, 24h forecast |
| `05_feature_based_models` | Feature construction and leakage checks, then the model |
| `06_foundation_model` | Chronos zero-shot, interval calibration, model-size comparison |
| `07_model_comparison` | All models, skill scores, error diagnostics |

Saved with outputs, so they read without executing. To re-run one:
`jupyter nbconvert --to notebook --execute --inplace notebooks/07_model_comparison.ipynb`

## Tests

```bash
pytest
```

97 tests covering the metrics (MASE is zero for a perfect forecast; MAE and MASE skill scores agree),
benchmark correctness, the horizon constraints on lag and rolling features, the rolling-origin
backtest (a probe asserts no origin ever sees future data), SARIMAX order selection and convergence
handling, feature-group ablation, and the foundation model interface — the last using a stub, so the
suite runs without downloading weights.

## Report

`Assignment report.pdf` — dataset, preprocessing, modelling, evaluation design, results and error
analysis, the six assignment questions, limitations, and a final recommendation.
