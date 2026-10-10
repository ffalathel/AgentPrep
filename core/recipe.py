"""Fitted preprocessing recipe: what Levels 2 and 4 learned, replayable on new data.

Every cleaning action and engineered feature is split into fit (learn parameters
from the training data) and apply (use them). The pipeline applies its own fitted
steps, so ``apply_recipe(training_data)`` reproduces the training output exactly,
and new data is transformed with training statistics, never its own.
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from level1_ingestion.normalizer import safe_cast_types

RECIPE_VERSION = 1


class RecipeError(ValueError):
    """Raised when a step cannot be fit or a recipe cannot be applied."""


def _py(value: Any) -> Any:
    """numpy scalar -> plain Python so the recipe is JSON-serializable."""
    return value.item() if isinstance(value, np.generic) else value


def _iqr_bounds(s: pd.Series) -> tuple[float, float]:
    q25, q75 = s.quantile(0.25), s.quantile(0.75)
    iqr = q75 - q25
    return _py(q25 - 1.5 * iqr), _py(q75 + 1.5 * iqr)


def placeholder_mask(s: pd.Series, value: Any) -> pd.Series:
    """Where ``s`` holds ``value`` (text compared without surrounding spaces)."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s == value if isinstance(value, (int, float)) and not isinstance(value, bool) else pd.Series(False, index=s.index)
    return s.notna() & (s.astype(str).str.strip() == str(value).strip())


def fit_step(
    op: str,
    df: pd.DataFrame,
    columns: list[str],
    name: Optional[str] = None,
    method: Optional[str] = None,
    value: Any = None,
) -> dict[str, Any]:
    """Learn the parameters for one step from ``df``."""
    step: dict[str, Any] = {"op": op, "columns": list(columns)}
    if name:
        step["name"] = name
    s = df[columns[0]]

    if op == "replace_with_missing":
        step["value"] = _py(value)
    elif op == "impute":
        if method == "mean":
            value = s.mean()
        elif method == "median":
            value = s.median()
        elif method == "mode":
            modes = s.mode()
            value = modes.iloc[0] if len(modes) else None
        else:
            raise RecipeError(f"Unknown imputation method: {method}")
        step.update(method=method, value=_py(value))
    elif op in ("clip", "flag"):
        if not pd.api.types.is_numeric_dtype(s):
            raise RecipeError(f"Cannot {op} non-numeric column: {columns[0]}")
        step["lower"], step["upper"] = _iqr_bounds(s)
    elif op == "standard_scaler":
        step.update(mean=_py(s.mean()), std=_py(s.std()))
    elif op == "min_max_scaler":
        step.update(min=_py(s.min()), max=_py(s.max()))
    elif op in ("log", "log10"):
        step["min"] = _py(s.min())
    elif op == "bin_uniform":
        _, edges = pd.cut(s, bins=10, retbins=True, duplicates="drop")
        step["edges"] = [float(e) for e in edges]
    elif op == "bin_quantile":
        _, edges = pd.qcut(s, q=10, retbins=True, duplicates="drop")
        step["edges"] = [float(e) for e in edges]
    elif op == "one_hot_encode":
        step["categories"] = [_py(c) for c in sorted(s.dropna().unique(), key=str)]
    elif op == "frequency_encode":
        step["counts"] = [[_py(k), int(c)] for k, c in s.value_counts().items()]
    return step


def apply_step(df: pd.DataFrame, step: dict[str, Any], raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Apply one fitted step. ``raw`` is the data before cleaning (for missing indicators)."""
    op, cols = step["op"], step["columns"]
    col, out = cols[0], step.get("name")
    s = df[col]

    if op == "drop":
        return df.drop(columns=[col])
    if op == "replace_with_missing":
        df[col] = s.mask(placeholder_mask(s, step["value"]))
    elif op == "impute":
        if step["value"] is not None:
            df[col] = s.fillna(step["value"])
    elif op == "clip":
        df[col] = s.clip(lower=step["lower"], upper=step["upper"])
    elif op == "flag":
        df[f"{col}_outlier"] = (s < step["lower"]) | (s > step["upper"])
    elif op == "standard_scaler":
        std = step["std"]
        df[out] = 0.0 if not std or np.isnan(std) else (s - step["mean"]) / std
    elif op == "min_max_scaler":
        span = step["max"] - step["min"]
        df[out] = 0.0 if not span or np.isnan(span) else (s - step["min"]) / span
    elif op == "log":
        df[out] = np.log1p((s - step["min"]).clip(lower=0))
    elif op == "log10":
        df[out] = np.log10((s - step["min"]).clip(lower=0) + 1)
    elif op in ("bin_uniform", "bin_quantile"):
        # Open outer edges so values outside the training range land in the end bins
        edges = [-np.inf, *step["edges"][1:-1], np.inf]
        df[out] = pd.cut(s, bins=edges, labels=False)
    elif op == "one_hot_encode":
        dummies = {f"{out}_{cat}": s == cat for cat in step["categories"]}
        return pd.concat([df, pd.DataFrame(dummies, index=df.index)], axis=1)
    elif op == "frequency_encode":
        counts = s.map(dict(step["counts"]))
        df[out] = counts.fillna(0).where(s.notna())  # unseen category -> 0
    elif op == "extract_year":
        df[out] = pd.to_datetime(s).dt.year
    elif op == "extract_month":
        df[out] = pd.to_datetime(s).dt.month
    elif op == "extract_day_of_week":
        df[out] = pd.to_datetime(s).dt.dayofweek
    elif op == "extract_hour":
        df[out] = pd.to_datetime(s).dt.hour
    elif op == "multiply":
        df[out] = s * df[cols[1]]
    elif op == "divide":
        df[out] = (s / df[cols[1]].replace(0, np.nan)).fillna(0)
    elif op == "add":
        df[out] = s + df[cols[1]]
    elif op == "subtract":
        df[out] = s - df[cols[1]]
    elif op == "square":
        df[out] = s**2
    elif op == "missing_indicator":
        # Read missingness before imputation, or the indicator is always 0
        source = raw if raw is not None and col in raw.columns else df
        df[out] = source[col].isna().astype(int)
    else:
        raise RecipeError(f"Unknown step: {op}")
    return df


def mark_placeholders_missing(df: pd.DataFrame, steps: list[dict[str, Any]]) -> pd.DataFrame:
    """The data before cleaning, with placeholder values already read as missing (for missing indicators)."""
    df = df.copy()
    for step in steps:
        if step["op"] == "replace_with_missing":
            df = apply_step(df, step)
    return df


def build_recipe(
    column_names: dict[str, str], target: str, input_columns: list[str], steps: list[dict], output_columns: list[str]
) -> dict[str, Any]:
    return {
        "version": RECIPE_VERSION,
        "target": target,
        "column_names": column_names,
        "input_columns": input_columns,
        "steps": steps,
        "output_columns": output_columns,
    }


def apply_recipe(df: pd.DataFrame, recipe: dict[str, Any]) -> pd.DataFrame:
    """Transform raw data exactly as the fitted pipeline did. The target column is optional."""
    target = recipe["target"]
    df = safe_cast_types(df.rename(columns=recipe["column_names"]))
    missing = [c for c in recipe["input_columns"] if c not in df.columns and c != target]
    if missing:
        raise RecipeError(f"Input is missing columns the pipeline was fit on: {missing}")

    raw = mark_placeholders_missing(df, recipe["steps"])
    for step in recipe["steps"]:
        df = apply_step(df, step, raw)

    absent = [c for c in recipe["output_columns"] if c not in df.columns and c != target]
    if absent:
        raise RecipeError(f"Recipe did not produce columns: {absent}")
    return df[[c for c in recipe["output_columns"] if c in df.columns]]


def save_recipe(recipe: dict[str, Any], path: Path) -> None:
    Path(path).write_text(json.dumps(recipe, indent=2))


def load_recipe(path: str | Path) -> dict[str, Any]:
    try:
        recipe = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise RecipeError(f"Cannot read pipeline file {path}: {e}") from e
    if recipe.get("version") != RECIPE_VERSION:
        raise RecipeError(f"Unsupported pipeline version: {recipe.get('version')}")
    return recipe
