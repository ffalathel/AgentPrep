"""TEST_PLAN.md sections 4-7: quality profile, artifacts, exit codes."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from cli import run_pipeline
from level4_feature.generator import FeatureGenerationError, FeatureGenerator
from tests.conftest import StubLLM, synthetic_df


def test_5_1_duplicate_rows_are_counted(run):
    df = synthetic_df()
    df = pd.concat([df, df.head(7)], ignore_index=True)
    _, orch = run(df)
    assert orch.quality_profile.duplicate_row_count == 7


def test_5_2_imputation_fills_missing_values(run):
    df = synthetic_df()
    df.loc[:49, "num_a"] = np.nan
    llm = StubLLM(actions=[{"column": "num_a", "action": "impute", "method": "median"}])
    _, orch = run(df, llm=llm)
    assert orch.feature_dataframe["num_a"].isna().sum() == 0


def test_6_1_and_6_2_artifacts_include_parquet_csv_reports(run, tmp_path):
    run(synthetic_df())
    out = tmp_path / "out"
    assert list(out.glob("artifacts/*/dataset_*.parquet"))
    assert list(out.glob("artifacts/*/dataset_*.csv"))
    assert list(out.glob("reports/*/*.md"))


def test_6_3_every_engineered_column_has_provenance(run):
    llm = StubLLM(
        features=[
            {"name": "num_b_log", "source_columns": ["num_b"], "transformation": "log"},
            {"name": "a_times_b", "source_columns": ["num_a", "num_b"], "transformation": "multiply"},
        ]
    )
    _, orch = run(synthetic_df(), llm=llm)
    engineered = set(orch.feature_dataframe.columns) - {"num_a", "num_b", "cat_c", "target"}
    assert engineered == {"num_b_log", "a_times_b"}
    assert {p.feature_name for p in orch.feature_provenance} == engineered


def test_7_1_invalid_intent_exits_1(tmp_path):
    cfg = tmp_path / "intent.yaml"
    cfg.write_text(yaml.safe_dump({"dataset": {"path": "x.csv"}, "task": {"type": "astrology"}}))
    assert run_pipeline(str(cfg)) == 1


def test_7_2_unremediated_leak_exits_2(run):
    df = synthetic_df()
    df["final_review"] = df["target"]
    code, orch = run(df, data_quality={"auto_remediate": False})
    assert code == 2
    assert orch.governance_decision.leakage_detected


def test_7_3_runtime_error_exits_3(run, monkeypatch):
    def boom(self, features):
        raise FeatureGenerationError("disk on fire")

    monkeypatch.setattr(FeatureGenerator, "generate_features", boom)
    code, _ = run(synthetic_df())
    assert code == 3


def test_7_4_config_file_run_end_to_end(tmp_path):
    csv = tmp_path / "data.csv"
    synthetic_df().to_csv(csv, index=False)
    cfg = tmp_path / "intent.json"
    cfg.write_text(
        json.dumps(
            {
                "dataset": {"path": str(csv)},
                "task": {"type": "classification", "target_column": "target"},
                "model": {"family": "tree"},
                "preferences": {"outlier_policy": "preserve", "interpretability_priority": "high"},
                "constraints": {"max_features": 50, "max_interactions": 5, "max_cardinality": 1000},
            }
        )
    )
    assert run_pipeline(str(cfg), output_path=str(tmp_path / "out")) == 0


def test_4_1_class_imbalance_warns_without_resampling(run):
    df = synthetic_df()
    df["target"] = (df.index % 20 == 0).astype(int)  # 5% positives
    code, orch = run(df)
    assert code == 0
    assert any("Class imbalance" in w for w in orch.governance_decision.warnings)
    assert len(orch.feature_dataframe) == len(df)


def test_4_4_protected_attributes_warn(run):
    df = synthetic_df()
    df["race"], df["sex"] = ["a", "b"] * 400, ["m", "f"] * 400
    code, orch = run(df)
    assert code == 0
    assert "Protected attributes detected: race, sex" in orch.governance_decision.warnings


def test_4_5_multicollinearity_warns_by_default_and_removes_when_strict(run):
    df = synthetic_df()
    df["num_a_dup"] = df["num_a"] * 2 + 0.001
    _, orch = run(df)
    assert any("Multicollinearity" in w for w in orch.governance_decision.warnings)
    assert "num_a_dup" in orch.feature_dataframe.columns

    code, orch = run(df, data_quality={"warn_on_quality_issues": False})
    assert code == 0
    assert "num_a_dup" not in orch.feature_dataframe.columns


@pytest.mark.parametrize(
    "policy, action, status",
    [("preserve", "flag", "rejected"), ("preserve", "clip", "rejected"), ("clip", "clip", "applied"), ("flag", "flag", "applied"), ("flag", "clip", "rejected")],
)
def test_5_3_outlier_actions_follow_policy(run, policy, action, status):
    llm = StubLLM(actions=[{"column": "num_b", "action": action}])
    _, orch = run(synthetic_df(), llm=llm, prefs={"outlier_policy": policy})
    assert orch.applied_actions[0].status == status


def test_distribution_shift_only_reported_when_real(run):
    _, orch = run(synthetic_df())
    assert not any("Distribution shift" in w for w in orch.governance_decision.warnings)

    rng = np.random.default_rng(4)
    ts = pd.DataFrame({"level": np.r_[np.zeros(640), np.full(160, 5.0)] + rng.normal(size=800)})
    ts["value"] = rng.normal(size=800)
    _, orch = run(ts, target="value", task="time_series")
    assert any("Distribution shift" in w and "level" in w for w in orch.governance_decision.warnings)


def test_7_4_interactive_wizard_end_to_end(tmp_path, monkeypatch, capsys):
    import cli

    csv = tmp_path / "data.csv"
    synthetic_df().to_csv(csv, index=False)
    answers = iter(
        [
            "4",  # LLM provider: none
            str(csv),
            "",  # task type: classification (default)
            "4",  # target column: 4th column, "target"
            "",  # model family (default)
            "",  # use suggested constraints
            "",  # accept them
            "",  # outlier policy (default)
            "",  # allow column dropping (default)
            "",  # interpretability (default)
            "",  # proceed
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr("sys.argv", ["agentprep", "run", "--output", str(tmp_path / "out")])

    assert cli.main() == 0
    assert next(answers, None) is None  # every prompt was asked, none extra
    assert "Pipeline completed successfully" in capsys.readouterr().out
    assert list((tmp_path / "out").glob("artifacts/*/pipeline_*.json"))


def test_numbers_stored_as_text_with_blanks_become_numeric(run):
    # Telco TotalCharges: numeric, but blank cells make pandas read it as text
    df = synthetic_df()
    df["total"] = df["num_b"].round(2).astype(str)
    df.loc[:9, "total"] = " "
    llm = StubLLM(actions=[{"column": "total", "action": "impute", "method": "median"}])
    _, orch = run(df, llm=llm)
    assert orch.applied_actions[0].status == "applied"
    assert pd.api.types.is_numeric_dtype(orch.feature_dataframe["total"])
    assert orch.feature_dataframe["total"].isna().sum() == 0


def test_imputing_a_mostly_missing_column_is_rejected(run):
    df = synthetic_df()
    df.loc[:599, "num_b"] = np.nan  # 75% missing
    llm = StubLLM(actions=[{"column": "num_b", "action": "impute", "method": "median"}])
    _, orch = run(df, llm=llm)
    assert orch.applied_actions[0].status == "rejected"
    assert "75%" in orch.applied_actions[0].reason
