# Pairwise Leaks and Placeholder Missing Values Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two general gaps exposed by the held-out benchmark datasets: leaks that only appear when two columns are combined, and placeholder values (0, -1, 999, "?", ...) that stand in for "missing".

**Architecture:**
(1) The leak detector gains a pairwise check: every pair of columns is scored jointly (combined groups, and the difference when both are numeric or dates; split year/month/day columns count as one date). A pair that predicts the target while neither column does alone is a joint leak. Data alone can't say which column of the pair is at fault, so: if the LLM independently suspected exactly one of them, that column is dropped (LLM nominates, data confirms); otherwise governance warns, naming both columns and the score.
(2) A general `replace_with_missing` cleaning action with a `value`. Which placeholders are meaningful (Bank `pdays = -1` = never contacted, Adult `capital_gain = 0`) is domain knowledge, so the profiler surfaces candidates, the LLM decides, and deterministic code validates and executes, the project's existing split.

**Tech Stack:** Python 3.10+, pandas, pytest. No new dependencies.

**Spec:** `TEST_PLAN.md` "What's still open" items 4 and 6; measurements below.

## Evidence (measured on all 10 bench training splits)

- Pairwise, without merging split dates: best legitimate pair 0.73 (Hotel), 0.71 (Boston `crim − rm`, R²). Hotel's leak doesn't separate.
- With `arrival_date_year/month/day_of_month` merged into one date: `reservation_status_date − arrival` scores 0.95 (difference) / 0.98 (sign). Above the existing 0.9 threshold, with margin over every legitimate pair.
- Cost: all pairs on Hotel's 95k rows took 36 s; other datasets ≤ 3 s.
- Placeholder candidates found in 6 of 10 datasets: Pima zeros, Adult `"?"`, Bank `pdays = -1` and `"unknown"`, Stroke `"Unknown"`, Hotel `adults = 0`, Boston `zn = 0`. Most are meaningful, which is why the LLM decides.

## Global Constraints

- Leak threshold stays `PREDICTIVE_SCORE_THRESHOLD = 0.9` for pairs too.
- The LLM alone never removes or rewrites a column; every action passes `ActionExecutor.validate_action`.
- `apply_recipe(train_raw, recipe).equals(train_out)` stays true on all 20 bench runs.
- Missing indicators and the missingness-leak check must see replaced placeholders as missing.
- `pytest` passes without an API key; the benchmark runs with `venv/bin/python`.

## Review Focus

1. A pair where one column already leaks alone → reported once by the single check, not again as a pair. Test in Task 1.
2. A joint leak with no LLM suspicion → warning naming both columns, nothing dropped, exit 0. Test in Task 1.
3. Unparseable dates (NaT), and split dates with impossible days or month names vs numbers → no crash, NaT. Test in Task 1.
4. A placeholder that appears again in new data → it becomes NaN and is then filled with the *training* value. Test in Task 2.
5. A placeholder value that isn't at the edge of a numeric column's range (e.g. `5` in the middle of the data) → rejected. Test in Task 2.

---

### Task 1: Pairwise (joint) leak check

**Files:**
- Modify: `level5_policy/leakage_detector.py`
- Modify: `level5_governance/gatekeeper.py` (warning for unattributed joint leaks)
- Test: `tests/test_leakage.py`

**Interfaces:**
- Consumes: `single_column_score`, `_encode` (same file); `_looks_like_dates` from `level1_ingestion/schema_inferencer.py`; `pipeline_output["suspected_leaks"]` (list of `{"column", "reason"}`, already present).
- Produces:
  - `assembled_dates(df) -> dict[str, pd.Series]`: for each prefix `p` with `{p}year`, `{p}month`, and `{p}day_of_month` or `{p}day`, a datetime series keyed `"{p}year/month/day"`. Month may be a number or an English name (full or 3-letter); invalid dates become NaT.
  - `joint_leaks(df, y, classification) -> list[tuple[str, str, float]]`: `(col_a, col_b, score)` for each leaking pair. `col_b` may be an assembled-date key.
  - `LeakageDetector.find_joint_leaks(pipeline_output) -> list[tuple[str, str, float]]`: joint leaks on `source_dataframe` that are *not* attributed (stored for governance).
  - `PAIR_SAMPLE_ROWS = 20_000`.

- [ ] **Step 1: Write the failing tests** in `tests/test_leakage.py`. Helper `hotel_like_df()`: `synthetic_df()` plus `arrival` (2020-01-01 + 0..700 days) split into `arrival_year`, `arrival_month` (English name), `arrival_day_of_month`; `status_date` = arrival − 1..200 days if `target == 1` else arrival + 0..10 days, as `YYYY-MM-DD` strings with 2% `"not a date"`. Helper `xor_df()`: `synthetic_df()` plus `a`, `b` random ints 0..9 with `target = (a + b) % 2` (joint leak, neither alone).

```python
def test_1_11_joint_leak_confirmed_by_llm_is_dropped(run):
    llm = StubLLM(suspected_leaks=[{"column": "status_date", "reason": "set at checkout"}])
    _, orch = run(hotel_like_df(), llm=llm)
    assert "status_date" in orch.leakage_dropped_columns
    assert not any(c.startswith("status_date") for c in orch.feature_dataframe.columns)
    assert not {"arrival_year", "arrival_month", "arrival_day_of_month"} & orch.leakage_dropped_columns

def test_1_11_unattributed_joint_leak_is_warned_not_dropped(run):
    code, orch = run(xor_df())
    assert code == 0 and {"a", "b"} <= set(orch.feature_dataframe.columns)
    assert any("'a'" in w and "'b'" in w and "together" in w for w in orch.governance_decision.warnings)

def test_1_11_single_column_leak_not_reported_as_pair():
    df = synthetic_df(); df["copy"] = df["target"]
    assert all("copy" not in pair[:2] for pair in joint_leaks(df.drop(columns="target"), df["target"], True))

def test_1_11_clean_pairs_are_not_flagged():
    df = synthetic_df()
    assert joint_leaks(df.drop(columns="target"), df["target"], True) == []

def test_assembled_dates_parse_names_numbers_and_bad_days():
    df = pd.DataFrame({"a_year": [2020, 2020, 2021], "a_month": ["July", "2", "Feb"], "a_day_of_month": [4, 30, 1]})
    d = assembled_dates(df)["a_year/month/day"]
    assert d[0] == pd.Timestamp("2020-07-04") and pd.isna(d[1]) and d[2] == pd.Timestamp("2021-02-01")
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_leakage.py -k "1_11 or assembled" -v`
Expected: FAIL (ImportError on `joint_leaks`, `assembled_dates`).

- [ ] **Step 3: Implement**
  - `joint_leaks`: sample up to `PAIR_SAMPLE_ROWS` rows (`random_state=0`). Candidates = columns that score < threshold alone, plus `assembled_dates`. Numeric view = numeric columns, plus date columns (`_looks_like_dates` or datetime64) as day ordinals, plus assembled dates. For each pair: `score = max(single_column_score(_encode(a) + "|" + _encode(b)), single_column_score(num[a] - num[b]) if both numeric)`. Keep pairs with `score >= PREDICTIVE_SCORE_THRESHOLD`; skip pairs made only of the parts of one assembled date with that date.
  - In `find_leaks`, after the existing source scan: run `joint_leaks` on `source_dataframe`. For each pair, `suspects = {a, b} & {s["column"] for s in suspected_leaks}`. If exactly one suspect is a real column, it joins `source_leaks` with the reason `"predicts the target together with '{other}' (held-out score {s:.2f}); also suspected by the LLM"`, so derived features are dropped by the existing provenance loop. Otherwise store the pair in `self.joint_leaks`.
  - `gatekeeper.py`: one warning per unattributed pair: `"Possible joint leak: '{a}' and '{b}' together predict the target (held-out score {s:.2f}); neither does alone. Check whether either is recorded after the outcome."`
  - Update the module docstring (check 4). Add a `ponytail:` comment: all pairs is O(columns²) on a 20k-row sample (~36 s → ~6 s on Hotel); triples are not checked.

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add level5_policy/leakage_detector.py level5_governance/gatekeeper.py tests/test_leakage.py
git commit -m "feat(leakage): detect leaks that only appear in a pair of columns"
```

---

### Task 2: `replace_with_missing` cleaning action

**Files:**
- Modify: `level2_quality/profiler.py` (placeholder candidates per column)
- Modify: `level2_quality/agent.py` (prompt)
- Modify: `level2_quality/executor.py` (validation, execution order)
- Modify: `core/recipe.py` (op, `mark_placeholders_missing`, `apply_recipe`)
- Modify: `core/orchestrator.py:387,426` (raw / source dataframe)
- Test: `tests/test_recipe.py`, `tests/test_validation.py`

**Interfaces:**
- Produces:
  - `ColumnQualityProfile.placeholder_candidates: list[dict]`: each entry is `{"value", "share"}`. Numeric: the min and the max when their share is ≥ 0.3% and the gap to the next distinct value is ≥ 0.5 × IQR. Text: values in `{"?", "-", "", "unknown", "n/a", "na", "null", "none", "missing"}` (case-insensitive, stripped). Limited to 5.
  - Recipe op `"replace_with_missing"` with `step["value"]`: `apply_step` does `df[col] = s.mask(s == value)` (for text, compares stripped values).
  - `mark_placeholders_missing(df, steps) -> pd.DataFrame`: a copy with only the `replace_with_missing` steps applied.

- [ ] **Step 1: Write the failing tests.** Frame: `synthetic_df()` plus `bmi` (normal 30±5, 20% set to 0), `visits` (Poisson(1), real zeros), and `job` (10% `"?"`).

```python
def test_5_8_placeholder_replaced_then_imputed(run):
    llm = StubLLM(actions=[
        {"column": "bmi", "action": "impute", "method": "median"},          # listed first on purpose
        {"column": "bmi", "action": "replace_with_missing", "value": 0},
        {"column": "job", "action": "replace_with_missing", "value": "?"},
    ], features=[{"name": "bmi_missing", "source_columns": ["bmi"], "transformation": "missing_indicator"}])
    code, orch = run(df, llm=llm)
    out = orch.feature_dataframe
    assert code == 0 and (out["bmi"] != 0).all() and "?" not in set(out["job"].dropna())
    assert out["bmi_missing"].tolist() == (df["bmi"] == 0).astype(int).tolist()
    assert set(out["bmi"][df["bmi"] == 0]) == {df.loc[df.bmi != 0, "bmi"].median()}

def test_5_8_replay_uses_training_fill_on_new_placeholders(run):
    # apply_recipe(train) equals the output; a new row with bmi=0 gets the training median and bmi_missing=1

def test_5_8_replace_with_missing_rejected(run):
    # value absent from column; value 5 in middle of num_b's range; target column; missing "value" -> all "rejected", df unchanged

def test_5_8_profiler_lists_placeholder_candidates():
    p = profile_column(df, "bmi"); assert {"value": 0.0, "share": 0.2} in p.placeholder_candidates  # share rounded to 3
    assert profile_column(df, "job").placeholder_candidates[0]["value"] == "?"
    assert profile_column(df, "num_b").placeholder_candidates == []

def test_5_8_prompt_offers_replace_with_missing():
    # "replace_with_missing" and "placeholder_candidates" both appear in the L2 prompt
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest -q -k 5_8`
Expected: FAIL.

- [ ] **Step 3: Implement**
  - `profiler.py`: `placeholder_candidates` as defined above.
  - `agent.py`: add `placeholder_candidates` to each column's info; add `replace_with_missing` to the actions, with `"value"` in the JSON format; add the rule: `- Use "replace_with_missing" with "value" when a value stands in for missing data (e.g. 0 for blood pressure or BMI, "?" for occupation). Keep values that mean something real (0 purchases, -1 for "never contacted", "Unknown" as an answer). Follow it with "impute" or rely on a missing indicator as usual`.
  - `executor.py`: valid only if `"value"` is given and occurs in the column. For numeric columns the value must equal the column's min or max. Not allowed on the target. In `execute_actions`, stable-sort `replace_with_missing` first (guard non-dict proposals). Keep the existing >50% imputation rule; it now counts replaced values (Pima `insulin`, ~49%, still passes).
  - `recipe.py`: the op; `apply_recipe` uses `raw = mark_placeholders_missing(df, recipe["steps"])`.
  - `orchestrator.py`: after L2, `self.pre_cleaning_dataframe = mark_placeholders_missing(self.normalized_dataframe, self.quality_steps)`; use it for `raw=` (line 387) and `"source_dataframe"` (line 426).

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add core/ level2_quality/ tests/
git commit -m "feat(quality): replace_with_missing for placeholder values, chosen by LLM, validated by code"
```

---

### Task 3: Re-benchmark and update the docs

**Files:** `bench/RESULTS.md`, `bench/results.json` (generated), `README.md`, `TEST_PLAN.md`

- [ ] **Step 1: Run** `venv/bin/python bench/run_bench.py` (about 10 minutes, real LLM calls). Pass criteria:
  - 20 runs, all exit 0.
  - The saved pipeline reproduces its training output exactly in all 20 runs.
  - No new leak flags on datasets without documented leaks; any joint-leak warnings there are listed and investigated.
  - Pipeline runtime on Hotel grows by < 15 s.
- [ ] **Step 2: Record, don't tune.** Report the outcome for every documented leak, which placeholders the LLM replaced on each dataset (expected: Pima's physiological zeros and Adult's `"?"`, but not Bank `pdays = -1` or Adult `capital_gain = 0`), and the score changes. Whatever comes out is the result; don't adjust thresholds to the benchmark.
- [ ] **Step 3: Update docs.**
  - `TEST_PLAN.md`: add rows 1.11 (joint leaks) and 5.8 (placeholders); refresh the before/after table and row 9.1; replace open items 4 and 6 with what's left (triples aren't checked; unattributed pairs are warnings only).
  - `README.md`: add the Level 2 and Level 5 bullets; update the Benchmark summary.
- [ ] **Step 4: Commit**

```bash
git add bench/RESULTS.md bench/results.json README.md TEST_PLAN.md
git commit -m "docs: re-benchmark after joint-leak check and placeholder handling"
```
