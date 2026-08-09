# scripts/run_pipeline.py
#
# Main pipeline entry point. Reproduces the full analysis from a fresh
# clone: load/prepare data, fit every model class, evaluate, and save
# forecasts, metrics, and figures under outputs/.
#
# Usage:
#   python scripts/run_pipeline.py

from appliance_energy.pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline()
