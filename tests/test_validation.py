"""TEST_PLAN.md sections 2-3: proposals are validated; LLM client behaves."""

import json
from types import SimpleNamespace

import pytest

from level2_quality.agent import AgentError, DataQualityAgent
from level4_feature.agent import FeatureAgentError, FeatureEngineeringAgent
from tests.conftest import StubLLM, synthetic_df
from utils.llm_client import LLMClientWrapper, resolve_model


class RawLLM:
    provider = "stub"

    def __init__(self, text):
        self.text = text

    def complete(self, prompt, **kwargs):
        return self.text


@pytest.mark.parametrize(
    "text",
    ["not json", "[]", '{"no_actions": []}', '{"actions": "drop everything"}', "```json\n{oops\n```"],
)
def test_2_2_malformed_quality_responses_raise(text):
    with pytest.raises(AgentError):
        DataQualityAgent(RawLLM(text)).propose_actions(*_l2_inputs())


@pytest.mark.parametrize(
    "text",
    ['{"actions": []}\n\nNote: no cleaning needed.', 'Here you go:\n```json\n{"actions": []}\n```\nDone.'],
)
def test_prose_around_json_is_tolerated(text):
    # Seen live: the model appended text after the JSON and the whole proposal was lost
    assert DataQualityAgent(RawLLM(text)).propose_actions(*_l2_inputs()) == {"actions": []}


@pytest.mark.parametrize(
    "features",
    [
        "x",
        [{"name": "a"}],
        [{"name": "a", "source_columns": "num_a", "transformation": "log"}],
        [{"name": "a", "source_columns": ["num_a"], "transformation": 1}],
    ],
)
def test_2_2_malformed_feature_responses_raise(features):
    with pytest.raises(FeatureAgentError):
        FeatureEngineeringAgent(RawLLM(json.dumps({"features": features}))).propose_features(*_l4_inputs())


def test_2_2_hostile_proposals_are_rejected_not_applied(run):
    llm = StubLLM(
        actions=[
            {"column": "target", "action": "drop"},
            {"column": "num_a; DROP TABLE", "action": "drop"},
            {"column": "num_a", "action": "exec", "method": "os.system"},
            {"column": "num_a", "action": "impute", "method": "__import__('os')"},
            {"action": "drop"},
        ],
        features=[
            {"name": "evil", "source_columns": ["num_a"], "transformation": "eval"},
            {"name": "ghost", "source_columns": ["missing_col"], "transformation": "log"},
            {"name": "too_many", "source_columns": ["num_a", "num_b", "cat_c"], "transformation": "multiply"},
        ],
    )
    code, orch = run(synthetic_df(), llm=llm)
    assert code == 0
    assert all(a.status == "rejected" for a in orch.applied_actions)
    assert len(orch.applied_actions) == 5
    assert orch.validated_features == []
    assert list(orch.feature_dataframe.columns) == ["num_a", "num_b", "cat_c", "target"]


def test_4_3_drop_rejected_when_dropping_disallowed(run):
    llm = StubLLM(actions=[{"column": "num_b", "action": "drop"}])
    _, orch = run(synthetic_df(), llm=llm, prefs={"allow_column_dropping": False})
    assert orch.applied_actions[0].status == "rejected"
    assert "num_b" in orch.feature_dataframe.columns


def test_2_4_replay_is_byte_identical(run, tmp_path):
    llm = StubLLM(
        actions=[{"column": "num_a", "action": "flag"}],
        features=[{"name": "num_b_log", "source_columns": ["num_b"], "transformation": "log"}],
    )
    _, first = run(synthetic_df(), llm=llm)
    _, second = run(synthetic_df(), llm=llm)
    assert first.feature_dataframe.equals(second.feature_dataframe)


def test_3_1_anthropic_thinking_block_is_skipped():
    blocks = [SimpleNamespace(type="thinking", thinking="..."), SimpleNamespace(type="text", text='{"actions": []}')]
    fake = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(content=blocks)))
    assert LLMClientWrapper(fake, "anthropic").complete("hi") == '{"actions": []}'


def test_3_4_model_is_configurable(monkeypatch):
    seen = {}

    def create(**kw):
        seen.update(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

    client = LLMClientWrapper(SimpleNamespace(messages=SimpleNamespace(create=create)), "anthropic")
    monkeypatch.setenv("AGENTPREP_MODEL", "my-model")
    client.complete("hi")
    assert seen["model"] == "my-model"
    monkeypatch.delenv("AGENTPREP_MODEL")
    assert resolve_model("anthropic") == "claude-sonnet-5-5"


def test_3_5_llm_failure_is_reported(run):
    code, orch = run(synthetic_df(), llm=StubLLM(fail=True))
    failures = [w for w in orch.governance_decision.warnings if "LLM agent failed" in w]
    assert len(failures) == 2  # Level 2 and Level 4


def test_3_6_stub_mode_runs(run):
    code, orch = run(synthetic_df(), llm=None)
    assert code == 0
    assert orch.governance_decision.warnings == [] or all("LLM" not in w for w in orch.governance_decision.warnings)


def _l2_inputs():
    from level1_ingestion.schema_inferencer import infer_schema
    from level2_quality.profiler import profile_dataset
    from tests.conftest import make_intent

    df = synthetic_df()
    return profile_dataset(df), infer_schema(df), make_intent("unused.csv")


def _l4_inputs():
    profile, schema, intent = _l2_inputs()
    return schema, profile, intent
