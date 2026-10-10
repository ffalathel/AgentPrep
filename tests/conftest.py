"""Shared fixtures: a stub LLM and a small synthetic dataset. No API key needed."""

import json

import numpy as np
import pandas as pd
import pytest

from core.orchestrator import PipelineOrchestrator
from intent.schema import IntentSchema


class StubLLM:
    """Returns canned proposals per agent; optionally fails every call."""

    provider = "stub"

    def __init__(self, actions=(), features=(), suspected_leaks=(), fail=False):
        self.actions, self.features, self.suspected_leaks, self.fail = actions, features, suspected_leaks, fail

    def complete(self, prompt, **kwargs):
        if self.fail:
            raise RuntimeError("simulated provider outage")
        if "feature engineering assistant" in prompt:
            return json.dumps({"features": list(self.features)})
        return json.dumps({"actions": list(self.actions), "suspected_leaks": list(self.suspected_leaks)})


def synthetic_df(n=800, seed=0):
    """Binary target driven noisily by num_a, so no clean column predicts it near-perfectly."""
    rng = np.random.default_rng(seed)
    num_a = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "num_a": num_a,
            "num_b": rng.uniform(0, 100, size=n),
            "cat_c": rng.choice(["red", "green", "blue"], size=n),
        }
    )
    df["target"] = (num_a + rng.normal(scale=1.5, size=n) > 0).astype(int)
    return df


def make_intent(csv_path, target="target", task="classification", **preferences):
    prefs = {"outlier_policy": "flag", "allow_column_dropping": True, "interpretability_priority": "medium"}
    prefs.update(preferences.pop("prefs", {}))
    return IntentSchema(
        dataset={"path": str(csv_path)},
        task={"type": task, "target_column": target},
        model={"family": "tree"},
        preferences=prefs,
        constraints={"max_features": 200, "max_interactions": 10, "max_cardinality": 100000},
        **preferences,
    )


@pytest.fixture
def run(tmp_path):
    """run(df, llm=None, **intent_kwargs) -> (exit_code, orchestrator)."""

    def _run(df, llm=None, **intent_kwargs):
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        orch = PipelineOrchestrator(make_intent(csv, **intent_kwargs), output_path=tmp_path / "out", llm_client=llm)
        return orch.run(), orch

    return _run
