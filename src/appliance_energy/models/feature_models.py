"""
Feature-based regression models: XGBoost, histogram gradient boosting,
and random forest.

Forecast construction
---------------------
Every lag and rolling feature in the table is built to respect the
24-hour horizon (see `appliance_energy.features`): a feature attached to
row `t` uses target values from `t - 24` or earlier only. A prediction
for row `t` is therefore a genuine 24-step-ahead forecast, and the whole
test period can be predicted in one pass without any risk of using
unobserved data.

This is deliberately a little more conservative than the rolling-origin
scheme used for the benchmarks and SARIMAX. There, a block forecast from
origin `o` predicts `o+1 ... o+24`, so the first step of each block is
only one step ahead. Here every step is a full 24 ahead. The comparison
therefore slightly favours the other models, which is the safe direction
to err in.

Tree models need no feature scaling, which also sidesteps the common
leakage trap of fitting a scaler on the full dataset before splitting.
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from appliance_energy import config, features


def build_model(name="xgboost", random_state=config.RANDOM_STATE, **overrides):
    """
    Construct an unfitted regressor by name.

    Defaults are modest and identical in spirit across model types so
    that the comparison reflects the model family rather than one having
    been tuned harder than the others.
    """

    name = name.lower()

    if name == "xgboost":
        from xgboost import XGBRegressor

        params = dict(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
        params.update(overrides)

        return XGBRegressor(**params)

    if name in ("histgb", "hist_gradient_boosting"):
        params = dict(
            max_iter=600,
            learning_rate=0.03,
            max_leaf_nodes=31,
            random_state=random_state,
        )
        params.update(overrides)

        return HistGradientBoostingRegressor(**params)

    if name in ("rf", "random_forest"):
        params = dict(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        params.update(overrides)

        return RandomForestRegressor(**params)

    raise ValueError(
        f"Unknown model '{name}'. Choose from: xgboost, histgb, random_forest."
    )


def split_feature_table(table, target=config.TARGET, test_steps=config.TEST_STEPS,
                        feature_cols=None):
    """
    Split the feature table into X/y train and test.

    The final `test_steps` rows form the test period, matching the
    14-day hold-out used by every other model in the project.
    """

    if feature_cols is None:
        feature_cols = [c for c in table.columns if c != target]

    X = table[feature_cols]
    y = table[target]

    return (
        X.iloc[:-test_steps], y.iloc[:-test_steps],
        X.iloc[-test_steps:], y.iloc[-test_steps:],
    )


def fit_feature_model(X_train, y_train, name="xgboost",
                      random_state=config.RANDOM_STATE, **overrides):
    """Fit a feature-based regressor on the training split."""

    model = build_model(name, random_state=random_state, **overrides)
    model.fit(X_train, y_train)

    return model


def predict_feature_model(model, X_test, index=None, name="feature_model"):
    """Predict and return a named series aligned to the test index."""

    if index is None:
        index = X_test.index

    return pd.Series(model.predict(X_test), index=index, name=name)


# ------------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------------

def get_feature_importance(model, feature_names):
    """
    Extract feature importances as a sorted series.

    Falls back to permutation-free native importances where available;
    HistGradientBoostingRegressor exposes none, so it returns None and
    the caller should use permutation importance instead.
    """

    if hasattr(model, "feature_importances_"):
        return (
            pd.Series(model.feature_importances_, index=feature_names)
            .sort_values(ascending=False)
        )

    return None


def permutation_feature_importance(model, X, y, n_repeats=5,
                                   random_state=config.RANDOM_STATE):
    """
    Permutation importance, measured on held-out data.

    More trustworthy than split-count importances for correlated
    features, at the cost of being slower.
    """

    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state,
        scoring="neg_mean_absolute_error",
    )

    return (
        pd.Series(result.importances_mean, index=X.columns)
        .sort_values(ascending=False)
    )


def group_importance(importances, groups):
    """Aggregate per-feature importances into per-group totals."""

    rows = []

    for name, cols in groups.items():
        present = [c for c in cols if c in importances.index]

        if not present:
            continue

        rows.append({
            "group": name,
            "n_features": len(present),
            "total_importance": float(importances[present].sum()),
            "mean_importance": float(importances[present].mean()),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("total_importance", ascending=False)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------
# Feature-group ablation
# ------------------------------------------------------------------

def split_train_valid_test(table, target=config.TARGET,
                           test_steps=config.TEST_STEPS,
                           valid_steps=config.VALID_STEPS, feature_cols=None):
    """
    Three-way chronological split: train / validation / test.

    The validation block sits immediately before the test block, so
    feature-set and hyper-parameter choices can be made without ever
    consulting the test period. Selecting on the test set is explicitly
    called out as leakage in the assignment brief.
    """

    if feature_cols is None:
        feature_cols = [c for c in table.columns if c != target]

    X = table[feature_cols]
    y = table[target]

    n_test, n_valid = test_steps, valid_steps

    return {
        "X_train": X.iloc[:-(n_test + n_valid)],
        "y_train": y.iloc[:-(n_test + n_valid)],
        "X_valid": X.iloc[-(n_test + n_valid):-n_test],
        "y_valid": y.iloc[-(n_test + n_valid):-n_test],
        "X_test": X.iloc[-n_test:],
        "y_test": y.iloc[-n_test:],
    }


GROUP_ORDER = ["calendar", "lag", "rolling", "house", "indoor", "outdoor"]


def run_feature_group_ablation(table, evaluate_fn, target=config.TARGET,
                               test_steps=config.TEST_STEPS,
                               valid_steps=config.VALID_STEPS,
                               name="xgboost", random_state=config.RANDOM_STATE,
                               evaluate_on="valid"):
    """
    Fit the model on cumulative feature groups to see which ones help.

    Groups are added in order of how available they would be in
    deployment: calendar first (always known), then lag and rolling
    (known, horizon-respecting), then the measured sensor and weather
    variables (which make the forecast conditional).

    `evaluate_on="valid"` (the default) scores each candidate on the
    validation block, so the feature set can be chosen without touching
    the test period. `evaluate_on="test"` is available for reporting the
    same curve on the test set *after* selection, purely as a diagnostic.

    `evaluate_fn(y_true, y_pred)` should return a dict of metrics.
    """

    groups = features.feature_groups(table.columns, target=target)
    order = [g for g in GROUP_ORDER if g in groups]

    results = []
    cumulative = []

    for group_name in order:
        cumulative = cumulative + groups[group_name]

        split = split_train_valid_test(
            table, target=target, test_steps=test_steps,
            valid_steps=valid_steps, feature_cols=cumulative,
        )

        if evaluate_on == "valid":
            X_fit, y_fit = split["X_train"], split["y_train"]
            X_eval, y_eval = split["X_valid"], split["y_valid"]
        else:
            # Train on train+validation, then score on the test block.
            X_fit = pd.concat([split["X_train"], split["X_valid"]])
            y_fit = pd.concat([split["y_train"], split["y_valid"]])
            X_eval, y_eval = split["X_test"], split["y_test"]

        model = fit_feature_model(X_fit, y_fit, name=name,
                                  random_state=random_state)
        predictions = predict_feature_model(model, X_eval, index=y_eval.index)

        metrics = evaluate_fn(y_eval, predictions)
        metrics.update({
            "added_group": group_name,
            "n_features": len(cumulative),
            "known_at_origin": group_name in features.KNOWN_AT_ORIGIN,
            "evaluated_on": evaluate_on,
        })

        results.append(metrics)

    return pd.DataFrame(results)


def columns_for_groups(table, group_names, target=config.TARGET):
    """Collect the feature columns belonging to a set of group names."""

    groups = features.feature_groups(table.columns, target=target)

    return [c for name in group_names for c in groups.get(name, [])]


def select_groups_by_validation(ablation, metric="MASE"):
    """
    Pick the cumulative group set with the best validation metric.

    Returns the winning group names, in the order they were added.
    """

    best_row = ablation.loc[ablation[metric].idxmin()]
    cutoff = list(ablation["added_group"]).index(best_row["added_group"])

    return list(ablation["added_group"])[: cutoff + 1]


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def select_and_fit(table, evaluate_fn, target=config.TARGET,
                   test_steps=config.TEST_STEPS, valid_steps=config.VALID_STEPS,
                   candidates=None, metric="MASE", random_state=config.RANDOM_STATE,
                   verbose=True):
    """
    Choose the feature groups and model family on the validation block,
    then refit on train + validation.

    Both choices are made without ever scoring against the test period,
    which the brief specifically flags as a source of leakage. Returns
    the fitted model plus a record of what was chosen and why.
    """

    candidates = candidates or config.FEATURE_MODEL_CANDIDATES

    ablation = run_feature_group_ablation(
        table, evaluate_fn, target=target, test_steps=test_steps,
        valid_steps=valid_steps, name=candidates[0],
        random_state=random_state, evaluate_on="valid",
    )

    chosen_groups = select_groups_by_validation(ablation, metric=metric)
    chosen_cols = columns_for_groups(table, chosen_groups, target=target)

    if verbose:
        print(f"Feature groups chosen on validation: {chosen_groups} "
              f"({len(chosen_cols)} features)")

    split = split_train_valid_test(
        table, target=target, test_steps=test_steps,
        valid_steps=valid_steps, feature_cols=chosen_cols,
    )

    scores = {}
    for name in candidates:
        model = fit_feature_model(split["X_train"], split["y_train"],
                                  name=name, random_state=random_state)
        predictions = predict_feature_model(
            model, split["X_valid"], index=split["y_valid"].index)
        scores[name] = evaluate_fn(split["y_valid"], predictions)[metric]

        if verbose:
            print(f"  {name:15s} validation {metric} = {scores[name]:.3f}")

    best_name = min(scores, key=scores.get)

    if verbose:
        print(f"Model family chosen on validation: {best_name}")

    # Refit on everything before the test period.
    X_fit = pd.concat([split["X_train"], split["X_valid"]])
    y_fit = pd.concat([split["y_train"], split["y_valid"]])

    model = fit_feature_model(X_fit, y_fit, name=best_name, random_state=random_state)

    selection = {
        "chosen_groups": chosen_groups,
        "chosen_columns": chosen_cols,
        "model_family": best_name,
        "validation_scores": scores,
        "selection_metric": metric,
    }

    return model, selection, ablation, split


def save_model(model, path=None):
    """Pickle a fitted feature model."""

    path = path or config.FEATURE_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as handle:
        pickle.dump(model, handle)

    print(f"Saved feature model to {path}")

    return path


def load_model(path=None):
    """Load a pickled feature model."""

    path = path or config.FEATURE_MODEL_PATH

    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_selection(selection, path=config.FEATURE_MODEL_SELECTION_PATH):
    """Record which feature groups and model family the validation chose."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)

    print(f"Saved feature-model selection to {path}")

    return selection
