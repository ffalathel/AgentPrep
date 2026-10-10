"""The fitted pipeline (pipeline_<run_id>.json) can be reapplied to new data."""

import numpy as np
import pandas as pd
import pytest

from cli import apply_pipeline
from core.recipe import RecipeError, apply_recipe, load_recipe
from level1_ingestion.loader import load_dataset
from tests.conftest import StubLLM, synthetic_df

ONE_COL = ["standard_scaler", "min_max_scaler", "log", "log10", "bin_uniform", "bin_quantile", "square"]
TWO_COL = ["multiply", "divide", "add", "subtract"]
DATE = ["extract_year", "extract_month", "extract_day_of_week", "extract_hour"]

EVERY_TRANSFORM = StubLLM(
    actions=[
        {"column": "num_b", "action": "impute", "method": "median"},
        {"column": "num_a", "action": "flag"},
    ],
    features=[{"name": f"b_{t}", "source_columns": ["num_b"], "transformation": t} for t in ONE_COL]
    + [{"name": f"ab_{t}", "source_columns": ["num_a", "num_b"], "transformation": t} for t in TWO_COL]
    + [{"name": f"d_{t}", "source_columns": ["signup_date"], "transformation": t} for t in DATE]
    + [
        {"name": "c_onehot", "source_columns": ["cat_c"], "transformation": "one_hot_encode"},
        {"name": "c_freq", "source_columns": ["cat_c"], "transformation": "frequency_encode"},
        {"name": "b_missing", "source_columns": ["num_b"], "transformation": "missing_indicator"},
    ],
)


def training_df():
    df = synthetic_df()
    df["Signup Date"] = pd.date_range("2024-01-01", periods=len(df), freq="7h").astype(str)
    df.loc[::10, "num_b"] = np.nan
    return df


@pytest.fixture
def fitted(run, tmp_path):
    code, orch = run(training_df(), llm=EVERY_TRANSFORM)
    assert code == 0
    assert orch.rejected_features == [], [r.reason for r in orch.rejected_features]
    assert all(a.status == "applied" for a in orch.applied_actions)
    return orch, load_recipe(orch.recipe_path)


def test_replay_on_training_data_reproduces_the_run_exactly(fitted, tmp_path):
    orch, recipe = fitted
    replayed = apply_recipe(load_dataset(tmp_path / "data.csv"), recipe)
    pd.testing.assert_frame_equal(replayed, orch.feature_dataframe)


def test_new_data_uses_training_statistics(fitted):
    orch, recipe = fitted
    train_median = orch.normalized_dataframe["num_b"].median()
    new = pd.DataFrame(
        {
            "num_a": [0.1, 50.0],  # 50 is far outside the training range
            "num_b": [np.nan, -10.0],  # -10 is below the training minimum
            "cat_c": ["purple", "red"],  # purple was never seen
            "Signup Date": ["2030-06-01 13:00:00", "2024-02-01 00:00:00"],
        }
    )
    out = apply_recipe(new, recipe)

    assert "target" not in out.columns  # target is optional at prediction time
    assert out.loc[0, "num_b"] == train_median
    assert out.loc[0, "b_missing"] == 1 and out.loc[1, "b_missing"] == 0
    assert not out.loc[0, [c for c in out.columns if c.startswith("c_onehot_")]].any()
    assert out.loc[0, "c_freq"] == 0
    assert out.loc[1, "num_a_outlier"]
    assert out[["b_bin_uniform", "b_bin_quantile", "b_log", "b_log10"]].notna().all().all()
    assert out.loc[0, "d_extract_year"] == 2030


def test_missing_indicator_survives_imputation(fitted):
    orch, _ = fitted
    was_missing = orch.normalized_dataframe["num_b"].isna().astype(int)
    assert orch.feature_dataframe["num_b"].isna().sum() == 0
    assert orch.feature_dataframe["b_missing"].equals(was_missing)


def test_leak_dropped_in_training_is_dropped_on_new_data(run, tmp_path):
    df = synthetic_df()
    df["final_review"] = df["target"]
    run(df)
    _, orch = run(df)
    out = apply_recipe(df, load_recipe(orch.recipe_path))
    assert "final_review" not in out.columns


def test_missing_input_column_is_an_error(fitted):
    _, recipe = fitted
    with pytest.raises(RecipeError, match="num_a"):
        apply_recipe(training_df().drop(columns="num_a"), recipe)


def test_cli_apply(fitted, tmp_path):
    orch, _ = fitted
    new = tmp_path / "new.csv"
    training_df().drop(columns="target").head(50).to_csv(new, index=False)
    out = tmp_path / "new_prepped.parquet"

    assert apply_pipeline(str(orch.recipe_path), str(new), str(out)) == 0
    assert len(pd.read_parquet(out)) == 50
    assert apply_pipeline(str(orch.recipe_path), str(tmp_path / "nope.csv"), str(out)) == 3


@pytest.mark.parametrize("module", ["level2_quality", "level4_feature", "core", "cli"])
def test_packages_import_standalone(module):
    # Guards against the core.recipe <-> level2/level4 import cycle
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", f"import {module}"], check=True)
