"""Benchmark: how often does deterministic code overrule the LLM, and does it catch leaks?

Runs the full pipeline on public mirrors of well-known Kaggle datasets and reports:
- L2 cleaning actions: proposed / applied / rejected
- L4 features: proposed / validated / rejected
- Leaks: for columns with documented target leakage, whether they survive to the output
- Held-out score: AgentPrep fits on an 80% split; its saved pipeline transforms the 20% test split
  (tree and linear models, each with its own run)

Usage:
    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY / GEMINI_API_KEY
    python bench/run_bench.py           # optional: AGENTPREP_MODEL=<model id>; needs scikit-learn
"""

import io
import json
import logging
import os
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.orchestrator import PipelineOrchestrator  # noqa: E402
from core.recipe import apply_recipe, load_recipe  # noqa: E402
from intent.schema import IntentSchema  # noqa: E402
from level1_ingestion.loader import load_dataset  # noqa: E402
from utils.llm_client import get_llm_client, resolve_model  # noqa: E402

BENCH = Path(__file__).resolve().parent
DATA = BENCH / "data"
OUT = BENCH / "out"

ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def load_titanic() -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_get("https://hbiostat.org/data/repo/titanic3.csv")))


def load_bank() -> pd.DataFrame:
    outer = zipfile.ZipFile(io.BytesIO(_get("https://archive.ics.uci.edu/static/public/222/bank+marketing.zip")))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
    return pd.read_csv(io.BytesIO(inner.read("bank-full.csv")), sep=";")


def load_telco() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    return pd.read_csv(io.BytesIO(_get(url)))


def load_adult_planted() -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    df = pd.read_csv(io.BytesIO(_get(url)), header=None, names=ADULT_COLS, skipinitialspace=True)
    # Planted leak: an innocuously named exact copy of the target. The most blatant case possible.
    df["final_review"] = df["income"]
    return df


def load_boston() -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_get("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")))


def load_hotels() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
    return pd.read_csv(io.BytesIO(_get(url)))


def load_hr_attrition() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
    return pd.read_csv(io.BytesIO(_get(url)))


def load_pima() -> pd.DataFrame:
    cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree", "age", "outcome"]
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    return pd.read_csv(io.BytesIO(_get(url)), header=None, names=cols)


def load_stroke() -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_get("https://raw.githubusercontent.com/fenago/datasets/main/healthcare-dataset-stroke-data.csv")))


def load_california() -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_get("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv")))


# name, loader, target, task, documented leak columns (original names)
DATASETS = [
    ("titanic", load_titanic, "survived", "classification", ["boat", "body"]),  # lifeboat no. / body id: post-outcome
    ("bank_marketing", load_bank, "y", "classification", ["duration"]),  # UCI docs: unknown before the call ends
    ("adult_planted", load_adult_planted, "income", "classification", ["final_review"]),
    ("telco_churn", load_telco, "Churn", "classification", []),
    ("boston_housing", load_boston, "medv", "regression", []),
    # Held-out set: not used to calibrate leak thresholds
    ("hotel_bookings", load_hotels, "is_canceled", "classification", ["reservation_status", "reservation_status_date"]),  # set at checkout/cancel
    ("hr_attrition", load_hr_attrition, "Attrition", "classification", []),
    ("pima_diabetes", load_pima, "outcome", "classification", []),
    ("stroke", load_stroke, "stroke", "classification", []),
    ("california_housing", load_california, "median_house_value", "regression", []),
]


class CountingClient:
    """Wraps the LLM client to count calls and failures per dataset."""

    def __init__(self, inner):
        self.inner, self.provider = inner, inner.provider
        self.calls = self.failures = 0

    def complete(self, prompt, **kw):
        self.calls += 1
        try:
            return self.inner.complete(prompt, **kw)
        except Exception:
            self.failures += 1
            raise


def make_intent(csv_path: Path, target: str, task: str, family: str) -> IntentSchema:
    return IntentSchema(
        dataset={"path": str(csv_path)},
        task={"type": task, "target_column": target},
        model={"family": family},
        preferences={"outlier_policy": "flag", "allow_column_dropping": True, "interpretability_priority": "medium"},
        constraints={"max_features": 200, "max_interactions": 10, "max_cardinality": 100000},
    )


def split(df: pd.DataFrame, target: str, task: str):
    from sklearn.model_selection import train_test_split

    df = df[df[target].notna()].reset_index(drop=True)
    strat = df[target] if task == "classification" else None
    return train_test_split(df, test_size=0.2, random_state=0, stratify=strat)


def holdout_score(train: pd.DataFrame, test: pd.DataFrame, target: str, task: str, family: str) -> float:
    """Fit a model on train, score on test: ROC AUC (binary) or R^2.

    Both arms (raw data and AgentPrep output) go through the same minimal sklearn
    prep, fitted on train only: median imputation, plus ordinal codes (tree) or
    scaling and one-hot encoding (linear). The difference between arms is AgentPrep's.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    def xy(df):
        X = df.drop(columns=target)
        for c in X.columns:
            if pd.api.types.is_bool_dtype(X[c]):
                X[c] = X[c].astype(float)
            elif not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype(str)
        return X, df[target]

    (X_tr, y_tr), (X_te, y_te) = xy(train), xy(test)
    num = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
    cat = [c for c in X_tr.columns if c not in num]
    if family == "tree":
        num_prep = SimpleImputer(strategy="median")
        cat_prep = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        num_prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        cat_prep = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30)
    prep = ColumnTransformer([("num", num_prep, num), ("cat", cat_prep, cat)])

    if task == "regression":
        model = HistGradientBoostingRegressor(random_state=0) if family == "tree" else Ridge()
        return float(r2_score(y_te, make_pipeline(prep, model).fit(X_tr, y_tr).predict(X_te)))
    labels = {v: i for i, v in enumerate(sorted(y_tr.unique(), key=str))}
    y_tr, y_te = y_tr.map(labels), y_te.map(labels)
    model = HistGradientBoostingClassifier(random_state=0) if family == "tree" else LogisticRegression(max_iter=3000)
    fitted = make_pipeline(prep, model).fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, fitted.predict_proba(X_te)[:, 1]))


def run_one(name, loader, target, task, leaks, family, llm):
    raw_path = DATA / f"{name}.csv"
    if not raw_path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        loader().to_csv(raw_path, index=False)
    train_path, test_path = DATA / f"{name}_train.csv", DATA / f"{name}_test.csv"
    for part, path in zip(split(load_dataset(raw_path), target, task), (train_path, test_path)):
        part.to_csv(path, index=False)
    train_raw, test_raw = load_dataset(train_path), load_dataset(test_path)

    orch = PipelineOrchestrator(make_intent(train_path, target, task, family), output_path=OUT / f"{name}_{family}", llm_client=llm)
    calls_before, fails_before = llm.calls, llm.failures
    exit_code = orch.run()

    actions = orch.applied_actions or []
    mapping = orch.column_name_mapping or {}
    final_cols = set(orch.feature_dataframe.columns) if orch.feature_dataframe is not None else set()
    dropped_by_llm = {a.column for a in actions if a.status == "applied" and a.action == "drop"}
    gov = orch.governance_decision
    suspected = {s["column"] for s in orch.suspected_leaks}
    target_norm = orch.normalized_target_column
    leak_norm = {mapping.get(c, c) for c in leaks}

    leak_rows = []
    for col in leaks:
        norm = mapping.get(col, col)
        if norm in dropped_by_llm:
            outcome = "removed by LLM drop action"
        elif norm in orch.leakage_dropped_columns:
            outcome = "caught by deterministic leak detector"
        elif norm in suspected:
            outcome = "surfaced as LLM-suspected leak (warning, kept)"
        elif norm in final_cols:
            outcome = "MISSED (still in output)"
        else:
            outcome = "absent from output (other reason)"
        leak_rows.append({"column": col, "outcome": outcome})

    # The fitted pipeline, applied to rows it never saw
    recipe = load_recipe(orch.recipe_path)
    train_out = orch.feature_dataframe
    test_out = apply_recipe(test_raw, recipe)
    replay_exact = apply_recipe(train_raw, recipe).equals(train_out)

    # Remove known leaks and anything derived from them, to measure feature value alone
    derived = {p.feature_name for p in orch.feature_provenance or [] if leak_norm & set(p.source_columns)}
    tainted = leak_norm | derived | {f"{c}_outlier" for c in leak_norm}
    clean = [c for c in train_out.columns if not any(c == t or c.startswith(f"{t}_") for t in tainted)]
    raw_tr, raw_te = (apply_recipe(d, {**recipe, "steps": [], "output_columns": recipe["input_columns"]}) for d in (train_raw, test_raw))
    raw_clean = [c for c in raw_tr.columns if c not in leak_norm]

    score = lambda tr, te: holdout_score(tr, te, target_norm, task, family)  # noqa: E731
    return {
        "dataset": name,
        "family": family,
        "rows": len(train_raw) + len(test_raw),
        "exit_code": exit_code,
        "llm_calls": llm.calls - calls_before,
        "llm_failures": llm.failures - fails_before,
        "l2_proposed": len(orch.proposed_actions or []),
        "l2_applied": sum(a.status == "applied" for a in actions),
        "l2_rejected": sum(a.status == "rejected" for a in actions),
        "l2_reject_reasons": Counter(a.reason.split(":")[0] for a in actions if a.status == "rejected"),
        "l4_proposed": len(orch.validated_features or []) + len(orch.rejected_features or []),
        "l4_validated": len(orch.validated_features or []),
        "l4_rejected": len(orch.rejected_features or []),
        "l4_reject_reasons": Counter(r.reason.split(":")[0].split("(")[0].strip() for r in orch.rejected_features or []),
        "l5_leak_flagged": sorted(orch.leakage_dropped_columns),
        "l5_approved": gov.approved if gov else None,
        "known_leaks": leak_rows,
        "suspected_leaks": orch.suspected_leaks,
        "replay_exact": replay_exact,
        "score_metric": "R2" if task == "regression" else "ROC AUC",
        "score_raw": score(raw_tr, raw_te),
        "score_raw_minus_known_leaks": score(raw_tr[raw_clean], raw_te[raw_clean]),
        "score_processed": score(train_out, test_out),
        "score_processed_minus_known_leaks": score(train_out[clean], test_out[clean]),
    }


def report(results) -> str:
    lines = [
        "| dataset | model family | LLM calls (failed) | L2 proposed → applied / rejected | L4 proposed → validated / rejected | L5 leak flags | replay exact |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['family']} | {r['llm_calls']} ({r['llm_failures']}) "
            f"| {r['l2_proposed']} → {r['l2_applied']} / {r['l2_rejected']} "
            f"| {r['l4_proposed']} → {r['l4_validated']} / {r['l4_rejected']} "
            f"| {', '.join(r['l5_leak_flagged']) or '-'} | {'yes' if r['replay_exact'] else 'NO'} |"
        )
    tot = lambda k: sum(r[k] for r in results)  # noqa: E731
    p2, p4 = tot("l2_proposed"), tot("l4_proposed")
    lines += [
        "",
        f"**Total LLM proposals: {p2 + p4}. Rejected by deterministic code: {tot('l2_rejected') + tot('l4_rejected')}** "
        f"(L2 {tot('l2_rejected')}/{p2}, L4 {tot('l4_rejected')}/{p4}).",
        "",
        "Rejection reasons (L2): " + json.dumps(sum((r["l2_reject_reasons"] for r in results), Counter())),
        "",
        "Rejection reasons (L4): " + json.dumps(sum((r["l4_reject_reasons"] for r in results), Counter())),
        "",
        "### Held-out model score",
        "",
        "AgentPrep runs on an 80% training split. Its saved pipeline (`pipeline_<run_id>.json`) is then applied "
        "to the untouched 20% test split, and a model trained on the training rows is scored on the test rows. "
        "Both arms get the same minimal sklearn prep fitted on train (median imputation; ordinal codes for trees, "
        "scaling and one-hot for linear models), so the gap between them is what AgentPrep adds.",
        "",
        "\"Known leaks removed\" drops the documented leak columns and any feature built from them, which isolates "
        "feature value from leak handling. Raw scores include leaks and are inflated where leaks exist.",
        "",
        "| dataset | model | metric | raw | raw, known leaks removed | AgentPrep | AgentPrep, known leaks removed | feature lift |",
        "|---|---|---|---|---|---|---|---|",
        *[
            f"| {r['dataset']} | {'gradient boosting' if r['family'] == 'tree' else 'linear'} | {r['score_metric']} "
            f"| {r['score_raw']:.3f} | {r['score_raw_minus_known_leaks']:.3f} | {r['score_processed']:.3f} "
            f"| {r['score_processed_minus_known_leaks']:.3f} "
            f"| {r['score_processed_minus_known_leaks'] - r['score_raw_minus_known_leaks']:+.3f} |"
            for r in results
        ],
        "",
        "### LLM-suspected leaks (surfaced as warnings, never auto-dropped)",
        "",
        *[
            f"- {r['dataset']} ({r['family']}): " + (", ".join(f"`{s['column']}` ({s['reason']})" for s in r["suspected_leaks"]) or "none")
            for r in results
        ],
        "",
        "### Documented leaks",
        "",
        "| dataset | model family | column | outcome |",
        "|---|---|---|---|",
    ]
    for r in results:
        for leak in r["known_leaks"]:
            lines.append(f"| {r['dataset']} | {r['family']} | {leak['column']} | {leak['outcome']} |")
    return "\n".join(lines)


def load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE loader; existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main():
    logging.disable(logging.WARNING)  # pipeline logs are noisy; the table is the output
    load_dotenv(ROOT / ".env")
    inner = get_llm_client()
    if inner is None:
        sys.exit("No LLM key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY.")
    llm = CountingClient(inner)
    model = resolve_model(llm.provider)
    print(f"provider={llm.provider} model={model}", file=sys.stderr)

    results = [run_one(*d, family, llm) for d in DATASETS for family in ("tree", "linear")]
    md = report(results)
    print(md)
    (BENCH / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (BENCH / "RESULTS.md").write_text(f"# Benchmark results\n\nprovider={llm.provider} model={model}\n\n{md}\n")


if __name__ == "__main__":
    main()
