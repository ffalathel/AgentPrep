"""Leakage detector for Level 5 policy enforcement.

Deterministic, data-driven checks on every column that reaches the output
(raw and engineered):
1. Target leakage - an engineered feature uses the target as input (provenance)
2. Predictive leakage - a single column predicts the target almost perfectly
   on held-out rows (exact copies, post-outcome fields)
3. Missingness leakage - whether a column is missing almost determines the
   target (fields only filled in for one outcome)
4. Joint leakage - two columns that predict the target together but not alone
   (e.g. a status date that falls before the arrival date only for cancellations).
   Data can't tell which of the pair is at fault: a column the LLM also suspected
   is dropped, otherwise the pair is reported as a warning

Semantic leaks (a column that is merely not known at prediction time) cannot be
proven from data alone; the LLM nominates those and governance surfaces them as
warnings (see GovernanceGatekeeper).

Temporal leakage: every catalog transformation is row-wise or a per-column
statistic, so no engineered feature can pull values from future rows. Raw
columns are covered by checks 2 and 3.
"""

import calendar
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd

from intent.schema import IntentSchema, TaskType
from level1_ingestion.schema_inferencer import _DATE_PREFIX

# ponytail: fixed thresholds calibrated on 5 public datasets (bench/run_bench.py):
# known leaks score >= 0.92, best legitimate column 0.68. Make configurable if real data disagrees.
PREDICTIVE_SCORE_THRESHOLD = 0.9
MISSINGNESS_PURITY_THRESHOLD = 0.97
# ponytail: every pair is O(columns^2) single-column scores on a row sample (Hotel, 33 columns: ~10 s);
# triples aren't checked. Pre-filter columns if wide datasets get slow.
PAIR_SAMPLE_ROWS = 20_000

_MONTHS = {name.lower(): i for names in (calendar.month_name, calendar.month_abbr) for i, name in enumerate(names) if name}


def _encode(series: pd.Series) -> pd.Series:
    """Discretize a column into groups: quantile bins for continuous numerics, values otherwise."""
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
        groups = pd.qcut(series.astype("float64"), 20, duplicates="drop").astype(str)
    else:
        groups = series.astype(str)
    return groups.where(series.notna(), "<missing>")


def is_classification(intent: IntentSchema, y: pd.Series) -> bool:
    return not (
        intent.task.type in (TaskType.REGRESSION, TaskType.TIME_SERIES)
        and pd.api.types.is_numeric_dtype(y)
    )


def single_column_score(x: pd.Series, y: pd.Series, classification: bool) -> float:
    """Held-out predictive power of one column alone.

    Classification: accuracy gain over the majority class, normalized to [0, 1].
    Regression: R^2 of per-group means. Unseen groups (e.g. IDs) fall back to the
    baseline, so identifiers score ~0.
    """
    train = np.random.default_rng(0).random(len(y)) < 0.5
    if train.all() or not train.any():
        return 0.0
    groups = _encode(x)
    if classification:
        y = y.astype(str)
        majority = y[train].mode()[0]
        lookup = pd.crosstab(groups[train], y[train]).idxmax(axis=1)
        pred = groups[~train].map(lookup).fillna(majority)
        base = (y[~train] == majority).mean()
        return float(((pred == y[~train]).mean() - base) / (1 - base)) if base < 1 else 0.0
    y = y.astype(float)
    pred = groups[~train].map(y[train].groupby(groups[train]).mean()).fillna(y[train].mean())
    total = ((y[~train] - y[~train].mean()) ** 2).sum()
    return float(1 - ((y[~train] - pred) ** 2).sum() / total) if total > 0 else 0.0


def missingness_purity(x: pd.Series, y: pd.Series) -> float:
    """Highest target purity among the missing / present groups of a column.

    Only counts groups covering >= 5% of rows whose dominant class is not already
    >90% of the data (otherwise purity says nothing).
    """
    missing = x.isna()
    if not 0.05 < missing.mean() < 0.95:
        return 0.0
    y = y.astype(str)
    best = 0.0
    for group in (missing, ~missing):
        if group.mean() < 0.05:
            continue
        shares = y[group].value_counts(normalize=True)
        if (y == shares.index[0]).mean() <= 0.9:
            best = max(best, float(shares.iloc[0]))
    return best


def _as_dates(series: pd.Series) -> Optional[pd.Series]:
    """Parsed dates if the column holds dates (unparseable values become NaT), else None."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if not (series.dtype == "object" or pd.api.types.is_string_dtype(series)):
        return None
    text = series.dropna().astype(str)
    if text.empty or text.head(1000).str.match(_DATE_PREFIX).mean() < 0.9:
        return None
    return pd.to_datetime(series, errors="coerce", format="mixed")


def assembled_dates(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Dates split across `<p>year`, `<p>month` (number or name), `<p>day_of_month` / `<p>day` columns."""
    out = {}
    for year in df.columns:
        if not str(year).endswith("year"):
            continue
        prefix = str(year)[: -len("year")]
        month = f"{prefix}month"
        day = next((c for c in (f"{prefix}day_of_month", f"{prefix}day") if c in df.columns), None)
        if month not in df.columns or day is None:
            continue
        m = df[month].astype(str).str.strip().str.lower()
        parts = pd.DataFrame({
            "year": pd.to_numeric(df[year], errors="coerce"),
            "month": m.map(_MONTHS).fillna(pd.to_numeric(m, errors="coerce")),
            "day": pd.to_numeric(df[day], errors="coerce"),
        })
        out[f"{prefix}year/month/day"] = pd.to_datetime(parts, errors="coerce")
    return out


def _days(dates: pd.Series) -> pd.Series:
    return (dates - pd.Timestamp("1970-01-01")).dt.days.astype("float64")


def joint_leaks(df: pd.DataFrame, y: pd.Series, classification: bool) -> list[tuple[str, str, float]]:
    """Pairs of columns that predict the target together while neither does alone.

    Each pair is scored on its combined values and, when both are numbers or dates,
    on their difference and its sign. Split year/month/day columns count as one date.
    """
    if len(y) > PAIR_SAMPLE_ROWS:
        y = y.sample(PAIR_SAMPLE_ROWS, random_state=0)
        df = df.loc[y.index]
    cols = [c for c in df.columns if single_column_score(df[c], y, classification) < PREDICTIVE_SCORE_THRESHOLD]
    groups = {c: _encode(df[c]) for c in cols}
    numeric = {}
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            numeric[c] = s.astype("float64")
        elif (dates := _as_dates(s)) is not None:
            numeric[c] = _days(dates)
    own_parts = {}
    for key, dates in assembled_dates(df).items():
        groups[key], numeric[key] = _encode(dates.astype(str).where(dates.notna())), _days(dates)
        prefix = key[: -len("year/month/day")]
        own_parts[key] = {prefix + part for part in ("year", "month", "day", "day_of_month")}

    found = []
    for a, b in combinations(groups, 2):
        if a in own_parts.get(b, ()) or b in own_parts.get(a, ()):
            continue  # a date and one of its own parts
        score = single_column_score(groups[a] + "|" + groups[b], y, classification)
        if a in numeric and b in numeric:
            gap = numeric[a] - numeric[b]
            score = max(score, single_column_score(gap, y, classification), single_column_score(np.sign(gap), y, classification))
        if score >= PREDICTIVE_SCORE_THRESHOLD:
            found.append((a, b, score))
    return found


class LeakageDetector:
    """Safety-critical leakage detector (deterministic)."""

    def __init__(self, intent: IntentSchema):
        self.intent = intent
        self.joint_leaks: list[tuple[str, str, float]] = []  # unattributed pairs, for governance warnings

    def find_leaks(self, pipeline_output: dict[str, Any]) -> dict[str, str]:
        """Return {column: reason} for every leaking column in the output.

        Scans the pre-cleaning data too (``source_dataframe``): cleaning such as
        imputation erases missingness leaks without removing the leak itself.
        """
        df = pipeline_output.get("feature_dataframe")
        target = pipeline_output.get("target_column")
        leaks: dict[str, str] = {}
        if target is None or df is None:
            return leaks

        source = pipeline_output.get("source_dataframe")
        source_leaks = self._scan(source, target) if source is not None else {}
        self.joint_leaks = []
        if source is not None and target in source.columns:
            known = source[target].notna()
            y = source.loc[known, target]
            suspected = {s["column"] for s in pipeline_output.get("suspected_leaks", [])}
            for a, b, score in joint_leaks(source.loc[known].drop(columns=target), y, is_classification(self.intent, y)):
                blamed = {a, b} & suspected & set(source.columns)
                if len(blamed) == 1:
                    col = blamed.pop()
                    other = b if col == a else a
                    source_leaks.setdefault(
                        col, f"predicts the target together with '{other}' (held-out score {score:.2f}); also suspected by the LLM"
                    )
                else:
                    self.joint_leaks.append((a, b, score))
        for col, reason in source_leaks.items():
            if col in df.columns:
                leaks[col] = f"{reason} (before cleaning)"

        for prov in pipeline_output.get("feature_provenance", []):
            sources = getattr(prov, "source_columns", [])
            if target in sources:
                leaks[prov.feature_name] = "uses the target column as input"
            elif bad := [c for c in sources if c in source_leaks]:
                leaks[prov.feature_name] = f"derived from leaking column(s) {bad}"

        for col, reason in self._scan(df, target).items():
            leaks.setdefault(col, reason)
        return {col: reason for col, reason in leaks.items() if col in df.columns}

    def _scan(self, df: pd.DataFrame, target: str) -> dict[str, str]:
        """Data-driven checks on every non-target column of one dataframe."""
        if target not in df.columns:
            return {}
        known = df[target].notna()
        y = df.loc[known, target]
        classification = is_classification(self.intent, y)
        found = {}
        for col in df.columns:
            if col == target:
                continue
            x = df.loc[known, col]
            score = single_column_score(x, y, classification)
            if score >= PREDICTIVE_SCORE_THRESHOLD:
                found[col] = f"predicts the target on its own (held-out score {score:.2f})"
            elif classification:
                purity = missingness_purity(x, y)
                if purity >= MISSINGNESS_PURITY_THRESHOLD:
                    found[col] = f"whether it is missing determines the target ({purity:.0%} pure)"
        return found

    def detect(self, pipeline_output: dict[str, Any]) -> bool:
        return bool(self.find_leaks(pipeline_output))

    def get_leakage_reason(self, pipeline_output: dict[str, Any]) -> str:
        leaks = self.find_leaks(pipeline_output)
        if not leaks:
            return "No leakage detected"
        return "; ".join(f"'{col}' {reason}" for col, reason in leaks.items())
