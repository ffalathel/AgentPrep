# Test Plan: README promises vs. reality

This plan lists what stopped AgentPrep from doing what the README says. Each claim gets a test that would prove it, and every test has a pass/fail outcome.

**Status key**
- ✅ **Passing**: covered by a test in `tests/` (stub LLM, no API key, runs in CI) or by `bench/run_bench.py` (real model)
- 🔧 **Fixed**: was broken; the fix and its test are noted
- ❓ **Open**: not verifiable here (see "What's still open")

**Run it:** `pytest` (76 tests, ~10 s, no key) and `python bench/run_bench.py` (needs an LLM key and scikit-learn).

## Before / after (benchmark, claude-sonnet-5-5)

| Metric | Before (2026-10-10, 5 datasets) | After (10 datasets, 5 of them unseen by calibration; 20 runs) |
|---|---|---|
| Documented leaks caught deterministically (Titanic `boat`, `body`; Bank `duration`; planted target copy; Hotel `reservation_status`, `reservation_status_date`) | 0 / 4 | **5 / 6** (`duration` surfaced as an LLM-suspected leak) |
| False leak flags (legitimate features deleted) | 8 | **0** |
| LLM calls failing silently | 7 / 10 | **0 / 40** (failures now reported) |
| LLM proposals rejected by deterministic code | 0 / 129 | 27 / 656 (mostly-missing imputations, nonexistent columns) |
| Pipeline reusable on new data | no | yes; exact replay on all 20 benchmark runs |
| Held-out score vs. standard sklearn prep (leaks removed) | not measured | linear gains on 5 of 10 (−0.011 to +0.098); trees −0.023 to +0.022 |
| Tests | none (`tests/` was gitignored) | 76, in CI on Python 3.10 and 3.12 |

Benchmark: 10 datasets × (tree, linear) runs, 80/20 split, AgentPrep fitted on train only. Leak outcomes were identical across every run so far. The second five datasets (Hotel Booking Demand, IBM HR Attrition, Pima Diabetes, Stroke Prediction, California Housing) were added after the fixes, to check that they generalize beyond the data they were calibrated on.

---

## 1. Leakage detection: *"Built-in policy enforcement and data leakage detection"*

| # | Status | Test | Notes |
|---|---|---|---|
| 1.1 | 🔧 | Planted exact copy of target is dropped | Detector used to skip raw columns entirely |
| 1.2 | 🔧 | Missingness leak (Titanic `body`) is dropped | New missingness-purity check |
| 1.3 | 🔧 | Near-perfect predictor (Titanic `boat`) is dropped | New held-out single-column score |
| 1.4 | 🔧 | Semantic leak (Bank `duration`) surfaced to user | LLM nominates; governance warns with measured score; never auto-dropped. LLM named all 4 known leaks in all three runs |
| 1.5 | 🔧 | Names containing `y_` / `label` not flagged | Name matching removed |
| 1.6 | 🔧 | Target `y` doesn't flag `day`, `pdays` | Same |
| 1.7 | ✅ | No cross-row transforms in catalog | Temporal leakage is impossible through engineered features; test fails if lag/rolling transforms are added without a temporal check |
| 1.8 | ✅ | Target as feature input rejected at L4 | Already worked |
| 1.9 | 🔧 | Imputation doesn't hide a leak | Found by the benchmark: the LLM imputed `boat`/`body`, erasing the signal. The detector now also scans pre-cleaning data and drops features derived from leaking columns |
| 1.10 | 🔧 | Regression target copy dropped; float16 columns don't crash | float16 crash found on Bank Marketing |
| 1.11 | 🔧 | Leaks that only appear in a pair of columns | Found on Hotel `reservation_status_date`: it barely predicts the target alone, but its gap to the (year/month/day split) arrival date does. Pairs are scored on combined values, difference and order; dropped when the LLM also suspected one column, otherwise warned. Bench: caught in both Hotel runs, no other pairs on 10 datasets |

Thresholds (`level5_policy/leakage_detector.py`) are calibrated on the 5 benchmark datasets: known leaks score ≥ 0.92, the best legitimate column 0.68.

## 2. Deterministic validation: *"All agent proposals are validated and executed deterministically"*

| # | Status | Test | Notes |
|---|---|---|---|
| 2.1 | ✅ | Hand-written bad proposals rejected | |
| 2.2 | ✅ | Malformed JSON, wrong types, hostile column names, non-catalog transforms | All rejected or raise; none applied |
| 2.3 | ✅ | Real-model rejection rate reported | 9 / 274 once the benchmark covered linear models and the imputation rule existed |
| 2.4 | ✅ | Replay with same proposals is byte-identical | README now says "given the same proposals" |
| 2.5 | ❓ | Live-LLM proposal stability | Proposal counts vary run to run (115–125); leak outcomes did not |

## 3. LLM providers

| # | Status | Test | Notes |
|---|---|---|---|
| 3.1 | 🔧 | Anthropic thinking block handled | `content[0].text` crashed on current models |
| 3.2 | 🔧 | OpenAI request/response through the real SDK | Fake HTTP transport, no key. Live model ID `gpt-4o` still unverified |
| 3.3 | 🔧 | Gemini request/response through the real SDK | Migrated from the deprecated `google-generativeai` to `google-genai`; fake HTTP transport, no key |
| 3.7 | 🔧 | Prose around the JSON reply tolerated | Seen live: the model appended text after its JSON and the whole feature proposal was lost. One shared parser for both agents |
| 3.4 | 🔧 | Model configurable via `AGENTPREP_MODEL` | Retired IDs were hard-coded in both agents |
| 3.5 | 🔧 | LLM failure reported in governance warnings | Used to be silent "0 proposals", exit 0 |
| 3.6 | ✅ | Stub mode (no key) runs | |

## 4. Governance & policy

| # | Status | Test | Notes |
|---|---|---|---|
| 4.1 | 🔧 | Class imbalance warns, no resampling | "Remediation" only logged; it now states why it doesn't resample (pre-split resampling leaks). Unused resampling module removed |
| 4.2 | 🔧 | Model-family rule | Always passed; removed |
| 4.3 | ✅ | Drop rejected when dropping disallowed | |
| 4.4 | ✅ | Protected attributes warned | |
| 4.5 | 🔧 | Multicollinearity warns by default, removed in strict mode | In default (warn) mode, quality issues were silently discarded, neither warned nor blocked |
| 4.6 | 🔧 | Distribution shift reported only when real | Random 80/20 split of i.i.d. data + no significance test flagged clean columns; strict mode rejected ordinary datasets |
| 4.7 | 🔧 | Governance fails closed | `except ImportError: return []` wrappers removed |

## 5. Data quality

| # | Status | Test | Notes |
|---|---|---|---|
| 5.1 | 🔧 | Duplicate rows counted | Profiler had no duplicate detection; now in profile, metadata, and L2 prompt |
| 5.2 | ✅ | Imputation fills missing values | |
| 5.3 | ✅ | Outlier actions follow policy | |
| 5.4 | 🔧 | Imputing a column >50% missing is rejected | Seen live: the LLM imputed Titanic `cabin` (77% missing) with its mode, erasing the "has a cabin" signal |
| 5.5 | 🔧 | Numbers stored as text (blank cells) become numeric | Telco `TotalCharges`; the LLM's median imputation was correctly rejected because the column was text |
| 5.6 | 🔧 | Non-snake_case column names work | Schema was inferred on raw names while cleaning used normalized ones, so proposals on e.g. `MonthlyCharges` could never apply |
| 5.7 | 🔧 | Date strings detected as datetime | CSV never yields a datetime dtype, so the four date transforms were unreachable |
| 5.8 | 🔧 | Placeholder values (0, -1, 999, "?") replaced with missing | Found on Pima (0 = not measured). Profiler lists candidates, LLM decides, code checks the value exists and, for numbers, is the column's min or max. Bench: the LLM's numeric values arrived as text ("0") and were all rejected until coerced; now applied on Pima, Adult, Bank, Stroke |

## 6. Artifacts

| # | Status | Test | Notes |
|---|---|---|---|
| 6.1 | 🔧 | Parquet dataset written | Never written. Also fixed: `numpy<2` pin + unpinned pyarrow made Parquet unusable on every fresh install |
| 6.2 | ✅ | JSON, CSV, Markdown reports written | |
| 6.3 | ✅ | Every engineered column has provenance | |
| 6.4 | 🔧 | Fitted pipeline saved and reusable on new data | There was no way to apply a run's cleaning and features to test or production data. Now `pipeline_<run_id>.json` + `agentprep apply` |
| 6.5 | 🔧 | Replaying the pipeline on the training file reproduces the output exactly | Every catalog transform, in tests and on all 20 benchmark runs |
| 6.6 | 🔧 | New data uses training statistics | Imputation fills, bounds, scalers, bins, categories; unseen categories and out-of-range values handled; target optional |
| 6.7 | 🔧 | Missing indicators survive imputation | Indicator was computed after imputation, so it was always 0 |

## 7. CLI, exit codes, packaging

| # | Status | Test | Notes |
|---|---|---|---|
| 7.1 | ✅ | Invalid intent → exit 1 | |
| 7.2 | ✅ | Unremediated leak → exit 2 | |
| 7.3 | 🔧 | Runtime error → exit 3 | L2/L4 errors returned 1 |
| 7.4 | ✅ | Config-file and interactive wizard runs end to end | Wizard driven through scripted `input()` |
| 7.6 | 🔧 | Packages import standalone | `import level2_quality` crashed on a circular import |
| 7.5 | 🔧 | Supported Python versions | Claimed 3.8+; code requires 3.10+. Now `requires-python >= 3.10`, tested on 3.10 and 3.12 |

## 8. Project hygiene

| # | Status | Notes |
|---|---|---|
| 8.1 | 🔧 | `tests/` was in `.gitignore`; README described tests that never existed. `pytest` now works from the repo root |
| 8.2 | 🔧 | Placeholder clone URL and issue links fixed |
| 8.3 | 🔧 | CI test workflow added (`.github/workflows/tests.yml`) |

## 9. Outcome

| # | Status | Notes |
|---|---|---|
| 9.1 | ✅ | Measured on a held-out split through the saved pipeline. Linear models gain on 5 of 10 datasets (Boston R² +0.098, California +0.029, Stroke +0.017), within −0.011 elsewhere. Gradient boosting −0.023 to +0.022. Where a semantic leak is kept as a warning (Bank `duration`), the score stays inflated until the user removes it |

---

## What's still open

1. **Live OpenAI and Gemini calls:** both paths are tested through the real SDKs against a fake server, but no live call has been made. Default model IDs (`gpt-4o`, `gemini-2.5-flash-lite`) are unverified.
2. **Trees:** features add little for gradient boosting, which already finds interactions and handles missing values. The value is for linear models.
3. **Benchmark variance:** one split per dataset. Titanic and Boston test sets are small; repeated splits would tighten the numbers.
4. **Leak thresholds:** calibrated on 5 datasets; constants in `level5_policy/leakage_detector.py`. On the 5 unseen datasets: 0 false flags. Only pairs of columns are checked, not groups of three or more.
5. **Temporal leakage:** safe only because no transform reads other rows (test 1.7). Time-series data isn't split by time.
6. **Placeholder judgement:** the LLM decides which placeholders mean missing. Debatable calls on the bench: Stroke `smoking_status = "Unknown"`, Titanic `fare = 0`.
7. **Dirty date columns:** Level 1 reads a date column with some junk values as text, so no date features are built from it.
8. **Benchmark run-to-run variance:** LLM proposals differ between runs. On Boston and HR Attrition, untouched by the new checks, the linear lift moved +0.120 → +0.098 and +0.006 → −0.007 between two full runs.
