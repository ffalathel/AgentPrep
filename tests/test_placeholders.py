"""TEST_PLAN.md 5.8: placeholder values (0, -1, 999, "?") that stand in for missing data."""

import numpy as np
import pandas as pd

from core.recipe import apply_recipe, load_recipe
from intent.schema import IntentSchema
from level1_ingestion.schema_inferencer import infer_schema
from level2_quality.agent import DataQualityAgent
from level2_quality.profiler import placeholder_candidates, profile_column, profile_dataset
from tests.conftest import StubLLM, make_intent, synthetic_df


def placeholder_df():
    df = synthetic_df()
    rng = np.random.default_rng(5)
    df["bmi"] = rng.normal(30, 5, len(df)).round(1)
    df.loc[rng.choice(len(df), 160, replace=False), "bmi"] = 0.0  # 20% recorded as 0 = not measured
    df["visits"] = rng.poisson(1, len(df))  # real zeros
    df["job"] = rng.choice(["clerk", "nurse", "chef"], len(df))
    df.loc[rng.choice(len(df), 80, replace=False), "job"] = "?"
    return df


PLACEHOLDER_LLM = StubLLM(
    actions=[
        {"column": "bmi", "action": "impute", "method": "median"},  # listed first on purpose
        {"column": "bmi", "action": "replace_with_missing", "value": 0},
        {"column": "job", "action": "replace_with_missing", "value": "?"},
    ],
    features=[{"name": "bmi_missing", "source_columns": ["bmi"], "transformation": "missing_indicator"}],
)


def test_5_8_placeholder_replaced_then_imputed(run):
    df = placeholder_df()
    code, orch = run(df, llm=PLACEHOLDER_LLM)
    out = orch.feature_dataframe
    assert code == 0
    assert (out["bmi"] != 0).all()
    assert "?" not in set(out["job"].dropna())
    assert out["bmi_missing"].tolist() == (df["bmi"] == 0).astype(int).tolist()
    filled = out.loc[df["bmi"] == 0, "bmi"]
    assert filled.nunique() == 1 and np.isclose(filled.iloc[0], df.loc[df["bmi"] != 0, "bmi"].median())  # float32
    assert (out["visits"] == 0).any()  # untouched


def test_5_8_replay_uses_training_fill_on_new_placeholders(run):
    df = placeholder_df()
    _, orch = run(df, llm=PLACEHOLDER_LLM)
    recipe = load_recipe(orch.recipe_path)
    assert apply_recipe(df, recipe).equals(orch.feature_dataframe)

    new = df.head(3).drop(columns="target").copy()
    new.loc[0, "bmi"], new.loc[1, "bmi"] = 0.0, 41.0
    out = apply_recipe(new, recipe)
    assert np.isclose(out.loc[0, "bmi"], df.loc[df["bmi"] != 0, "bmi"].median())
    assert out["bmi_missing"].tolist()[:2] == [1, 0]


def test_5_8_replace_with_missing_rejected(run):
    df = placeholder_df()
    mid_value = float(np.sort(df["num_b"])[400])
    llm = StubLLM(
        actions=[
            {"column": "bmi", "action": "replace_with_missing", "value": 999},  # not in the column
            {"column": "num_b", "action": "replace_with_missing", "value": mid_value},  # not an edge value
            {"column": "target", "action": "replace_with_missing", "value": 0},
            {"column": "bmi", "action": "replace_with_missing"},  # no value
        ]
    )
    _, orch = run(df, llm=llm)
    assert [a.status for a in orch.applied_actions] == ["rejected"] * 4
    assert orch.feature_dataframe["bmi"].eq(0).sum() == 160


def test_5_8_profiler_lists_placeholder_candidates():
    df = placeholder_df()
    assert {"value": 0.0, "share": 0.2} in profile_column(df, "bmi").placeholder_candidates
    assert profile_column(df, "job").placeholder_candidates == [{"value": "?", "share": 0.1}]
    assert profile_column(df, "num_b").placeholder_candidates == []


def test_5_8_prompt_offers_replace_with_missing(tmp_path):
    df = placeholder_df()
    intent: IntentSchema = make_intent(tmp_path / "unused.csv")
    prompt = DataQualityAgent()._construct_prompt(profile_dataset(df), infer_schema(df), intent)
    assert "replace_with_missing" in prompt
    assert "placeholder_candidates" in prompt


def test_5_8_common_placeholder_number_listed_even_when_frequent():
    # Like Pima `insulin`: half the rows are 0, so 0 isn't set apart by the spread of the data
    rng = np.random.default_rng(6)
    s = pd.Series(np.where(rng.random(800) < 0.49, 0, rng.lognormal(4.8, 0.6, 800).round()))
    assert {"value": 0.0, "share": round(float((s == 0).mean()), 3)} in placeholder_candidates(s)
