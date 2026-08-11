import numpy as np
import pandas as pd
import pytest

from appliance_energy import config, features
from appliance_energy.models import feature_models


@pytest.fixture(scope="module")
def table():
    """A small feature table with a learnable daily pattern."""
    index = pd.date_range("2020-01-06", periods=24 * 60, freq="h")
    rng = np.random.RandomState(0)

    hours = np.arange(len(index)) % 24
    target = 50 + 30 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 4, len(index))

    frame = pd.DataFrame(
        {
            config.TARGET: target,
            "T1": rng.normal(20, 1, len(index)),
            "RH_1": rng.normal(40, 2, len(index)),
            "T_out": rng.normal(5, 3, len(index)),
            "lights": rng.normal(10, 2, len(index)),
        },
        index=index,
    )

    return features.build_feature_table(frame, horizon=24)


def _metrics(y_true, y_pred):
    return {"MAE": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))}


# ------------------------------------------------------------------
# Model construction
# ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["xgboost", "histgb", "random_forest"])
def test_build_model_supports_each_family(name):
    model = feature_models.build_model(name)
    assert hasattr(model, "fit")


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        feature_models.build_model("not_a_model")


def test_build_model_accepts_overrides():
    model = feature_models.build_model("random_forest", n_estimators=7)
    assert model.n_estimators == 7


# ------------------------------------------------------------------
# Splitting
# ------------------------------------------------------------------

def test_split_feature_table_uses_final_rows_as_test(table):
    X_train, y_train, X_test, y_test = feature_models.split_feature_table(
        table, test_steps=48
    )

    assert len(X_test) == 48
    assert len(y_test) == 48
    assert len(X_train) == len(table) - 48
    # Test block must come strictly after the training block.
    assert X_train.index.max() < X_test.index.min()
    assert config.TARGET not in X_train.columns


def test_train_valid_test_split_is_chronological_and_disjoint(table):
    split = feature_models.split_train_valid_test(
        table, test_steps=48, valid_steps=48
    )

    assert len(split["X_valid"]) == 48
    assert len(split["X_test"]) == 48

    assert split["X_train"].index.max() < split["X_valid"].index.min()
    assert split["X_valid"].index.max() < split["X_test"].index.min()

    total = len(split["X_train"]) + len(split["X_valid"]) + len(split["X_test"])
    assert total == len(table)


def test_validation_block_sits_immediately_before_test(table):
    split = feature_models.split_train_valid_test(
        table, test_steps=48, valid_steps=48
    )

    gap = split["X_test"].index.min() - split["X_valid"].index.max()
    assert gap == pd.Timedelta(hours=1)


# ------------------------------------------------------------------
# Fitting and prediction
# ------------------------------------------------------------------

def test_prediction_length_and_index_match_the_test_block(table):
    X_train, y_train, X_test, y_test = feature_models.split_feature_table(
        table, test_steps=48
    )

    model = feature_models.fit_feature_model(
        X_train, y_train, name="random_forest", n_estimators=20
    )
    predictions = feature_models.predict_feature_model(model, X_test, index=y_test.index)

    assert len(predictions) == len(y_test)
    assert list(predictions.index) == list(y_test.index)
    assert predictions.notna().all()


def test_model_learns_the_daily_pattern_better_than_a_constant(table):
    X_train, y_train, X_test, y_test = feature_models.split_feature_table(
        table, test_steps=48
    )

    model = feature_models.fit_feature_model(
        X_train, y_train, name="random_forest", n_estimators=50
    )
    predictions = feature_models.predict_feature_model(model, X_test, index=y_test.index)

    model_mae = np.mean(np.abs(y_test - predictions))
    constant_mae = np.mean(np.abs(y_test - y_train.mean()))

    assert model_mae < constant_mae


def test_results_are_reproducible_with_a_fixed_seed(table):
    X_train, y_train, X_test, y_test = feature_models.split_feature_table(
        table, test_steps=48
    )

    def run():
        model = feature_models.fit_feature_model(
            X_train, y_train, name="random_forest", n_estimators=20, random_state=0
        )
        return feature_models.predict_feature_model(model, X_test, index=y_test.index)

    assert np.allclose(run().values, run().values)


# ------------------------------------------------------------------
# Importance
# ------------------------------------------------------------------

def test_feature_importance_covers_every_feature(table):
    X_train, y_train, _, _ = feature_models.split_feature_table(table, test_steps=48)

    model = feature_models.fit_feature_model(
        X_train, y_train, name="random_forest", n_estimators=20
    )
    importance = feature_models.get_feature_importance(model, list(X_train.columns))

    assert set(importance.index) == set(X_train.columns)
    assert (importance >= 0).all()
    assert importance.is_monotonic_decreasing


def test_histgb_reports_no_native_importance(table):
    X_train, y_train, _, _ = feature_models.split_feature_table(table, test_steps=48)

    model = feature_models.fit_feature_model(
        X_train, y_train, name="histgb", max_iter=10
    )

    assert feature_models.get_feature_importance(model, list(X_train.columns)) is None


def test_group_importance_totals_match_the_per_feature_sums(table):
    X_train, y_train, _, _ = feature_models.split_feature_table(table, test_steps=48)

    model = feature_models.fit_feature_model(
        X_train, y_train, name="random_forest", n_estimators=20
    )
    importance = feature_models.get_feature_importance(model, list(X_train.columns))
    groups = features.feature_groups(table.columns)

    grouped = feature_models.group_importance(importance, groups)

    assert grouped["total_importance"].sum() == pytest.approx(importance.sum())


# ------------------------------------------------------------------
# Ablation and selection
# ------------------------------------------------------------------

def test_ablation_adds_groups_cumulatively(table):
    ablation = feature_models.run_feature_group_ablation(
        table, _metrics, test_steps=48, valid_steps=48,
        name="random_forest", evaluate_on="valid",
    )

    # Feature counts must increase monotonically as groups are added.
    assert ablation["n_features"].is_monotonic_increasing
    assert ablation["added_group"].iloc[0] == "calendar"
    assert (ablation["evaluated_on"] == "valid").all()


def test_ablation_marks_availability_correctly(table):
    ablation = feature_models.run_feature_group_ablation(
        table, _metrics, test_steps=48, valid_steps=48,
        name="random_forest", evaluate_on="valid",
    )

    availability = dict(zip(ablation["added_group"], ablation["known_at_origin"]))

    assert availability["calendar"] is True or availability["calendar"] == True  # noqa: E712
    assert not availability["indoor"]
    assert not availability["outdoor"]


def test_selection_returns_the_prefix_up_to_the_best_score():
    ablation = pd.DataFrame({
        "added_group": ["calendar", "lag", "rolling", "indoor"],
        "MASE": [0.90, 0.66, 0.88, 1.40],
    })

    assert feature_models.select_groups_by_validation(ablation) == ["calendar", "lag"]


def test_selection_can_pick_the_first_group_only():
    ablation = pd.DataFrame({
        "added_group": ["calendar", "lag", "rolling"],
        "MASE": [0.50, 0.66, 0.88],
    })

    assert feature_models.select_groups_by_validation(ablation) == ["calendar"]


def test_columns_for_groups_returns_only_requested_groups(table):
    cols = feature_models.columns_for_groups(table, ["calendar", "lag"])
    groups = features.feature_groups(table.columns)

    assert set(cols) == set(groups["calendar"]) | set(groups["lag"])
    assert not any(c.startswith("roll_") for c in cols)
