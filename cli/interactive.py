"""Interactive CLI prompts for AgentPrep.

This module provides interactive prompts to collect user intent
without requiring a configuration file.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from intent.schema import (
    InterpretabilityPriority,
    ModelFamily,
    OutlierPolicy,
    TaskType,
)
from level1_ingestion.loader import DatasetLoadError, load_dataset
from utils import get_logger

from .constraint_advisor import ConstraintAdvisor

logger = get_logger(__name__)


def prompt_dataset_path() -> tuple[str, pd.DataFrame]:
    """Prompt user for dataset path and load it.

    Returns:
        Tuple of (dataset_path, dataframe)

    Raises:
        SystemExit: If user cancels or file cannot be loaded
    """
    while True:
        path = input("\nEnter path to your dataset (CSV or Parquet): ").strip()

        if not path:
            print("✗ Dataset path cannot be empty. Please try again.")
            continue

        try:
            df = load_dataset(path)
            print(f"✓ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            return path, df
        except DatasetLoadError as e:
            print(f"✗ Error loading dataset: {e}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != "y":
                print("Cancelled.")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != "y":
                print("Cancelled.")
                sys.exit(1)


def display_columns(columns: list[str]) -> None:
    """Display columns as a numbered list.

    Args:
        columns: List of column names
    """
    print("\nAvailable columns:")
    for i, col in enumerate(columns, 1):
        print(f"  {i}. {col}")


def prompt_target_column(columns: list[str]) -> str:
    """Prompt user to select target column from numbered list.

    Args:
        columns: List of column names

    Returns:
        Selected column name

    Raises:
        SystemExit: If user cancels
    """
    display_columns(columns)

    while True:
        try:
            choice = input(f"\nSelect target column (enter number 1-{len(columns)}): ").strip()

            if not choice:
                print("✗ Selection cannot be empty. Please try again.")
                continue

            index = int(choice) - 1

            if 0 <= index < len(columns):
                selected = columns[index]
                print(f"✓ Selected: {selected}")
                return selected
            else:
                print(f"✗ Invalid selection. Please enter a number between 1 and {len(columns)}.")
        except ValueError:
            print("✗ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            sys.exit(1)


def prompt_enum(
    question: str,
    options: list[tuple[str, str]],
    default: Optional[str] = None,
) -> str:
    """Prompt user to select from enum options.

    Args:
        question: Question to display
        options: List of (value, label) tuples
        default: Default value (optional)

    Returns:
        Selected enum value

    Raises:
        SystemExit: If user cancels
    """
    print(f"\n{question}")
    for i, (value, label) in enumerate(options, 1):
        marker = " [default]" if default and value == default else ""
        print(f"  [{i}] {label}{marker}")

    default_num = None
    if default:
        for i, (value, _) in enumerate(options, 1):
            if value == default:
                default_num = i
                break

    prompt_text = f"Enter choice [1-{len(options)}]"
    if default_num:
        prompt_text += f" [{default_num}]"
    prompt_text += ": "

    while True:
        try:
            choice = input(prompt_text).strip()

            if not choice and default:
                print(f"✓ Selected: {default}")
                return default

            index = int(choice) - 1

            if 0 <= index < len(options):
                selected_value, selected_label = options[index]
                print(f"✓ Selected: {selected_label}")
                return selected_value
            else:
                print(f"✗ Invalid selection. Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("✗ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            sys.exit(1)


def prompt_int(
    question: str,
    min_val: int,
    max_val: int,
    default: Optional[int] = None,
) -> int:
    """Prompt user for integer input with validation.

    Args:
        question: Question to display
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        default: Default value (optional)

    Returns:
        Validated integer value

    Raises:
        SystemExit: If user cancels
    """
    prompt_text = f"\n{question} ({min_val}-{max_val})"
    if default is not None:
        prompt_text += f" [default: {default}]"
    prompt_text += ": "

    while True:
        try:
            value = input(prompt_text).strip()

            if not value and default is not None:
                print(f"✓ Using default: {default}")
                return default

            int_value = int(value)

            if min_val <= int_value <= max_val:
                print(f"✓ Set to: {int_value}")
                return int_value
            else:
                print(f"✗ Value must be between {min_val} and {max_val}. Please try again.")
        except ValueError:
            print("✗ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            sys.exit(1)


def prompt_bool(question: str, default: bool = False) -> bool:
    """Prompt user for boolean input.

    Args:
        question: Question to display
        default: Default value

    Returns:
        Boolean value

    Raises:
        SystemExit: If user cancels
    """
    default_text = "y" if default else "n"
    prompt_text = f"\n{question} (y/n) [default: {default_text}]: "

    while True:
        try:
            value = input(prompt_text).strip().lower()

            if not value:
                print(f"✓ Using default: {'Yes' if default else 'No'}")
                return default

            if value in ("y", "yes"):
                print("✓ Selected: Yes")
                return True
            elif value in ("n", "no"):
                print("✓ Selected: No")
                return False
            else:
                print("✗ Invalid input. Please enter 'y' or 'n'.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            sys.exit(1)


def prompt_output_path() -> Optional[str]:
    """Prompt user for output directory path.

    Returns:
        Output path or None for default
    """
    path = input("\nOutput directory (press Enter for default): ").strip()
    if not path:
        return None
    return path


def display_summary(intent_dict: dict) -> None:
    """Display summary of collected intent.

    Args:
        intent_dict: Intent dictionary
    """
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"Dataset: {intent_dict['dataset']['path']}")
    print(f"Task Type: {intent_dict['task']['type']}")
    print(f"Target Column: {intent_dict['task']['target_column']}")
    print(f"Model Family: {intent_dict['model']['family']}")
    print(f"Outlier Policy: {intent_dict['preferences']['outlier_policy']}")
    print(f"Allow Column Dropping: {intent_dict['preferences']['allow_column_dropping']}")
    print(f"Interpretability Priority: {intent_dict['preferences']['interpretability_priority']}")
    print(f"Max Features: {intent_dict['constraints']['max_features']}")
    print(f"Max Interactions: {intent_dict['constraints']['max_interactions']}")
    print(f"Max Cardinality: {intent_dict['constraints']['max_cardinality']}")
    print("=" * 60)


def collect_intent_interactively() -> dict:
    """Collect user intent through interactive prompts.

    Returns:
        Intent dictionary compatible with IntentSchema

    Raises:
        SystemExit: If user cancels at any point
    """
    print("\n" + "=" * 60)
    print("Welcome to AgentPrep!")
    print("=" * 60)
    print("\nThis interactive wizard will guide you through configuring your preprocessing pipeline.")

    # Step 1: Dataset path and loading
    dataset_path, dataframe = prompt_dataset_path()

    # Step 2: Task type
    task_type = prompt_enum(
        "Select task type:",
        [
            (TaskType.CLASSIFICATION.value, "Classification"),
            (TaskType.REGRESSION.value, "Regression"),
            (TaskType.TIME_SERIES.value, "Time Series"),
            (TaskType.CLUSTERING.value, "Clustering"),
        ],
        default=TaskType.CLASSIFICATION.value,
    )

    # Step 3: Target column
    columns = list(dataframe.columns)
    target_column = prompt_target_column(columns)

    # Step 4: Model family
    model_family = prompt_enum(
        "Select model family:",
        [
            (ModelFamily.TREE.value, "Tree-based (Random Forest, XGBoost, etc.)"),
            (ModelFamily.LINEAR.value, "Linear (Logistic Regression, Linear Regression)"),
            (ModelFamily.NEURAL.value, "Neural Network"),
            (ModelFamily.UNKNOWN.value, "Unknown / Not sure"),
        ],
        default=ModelFamily.TREE.value,
    )

    # Step 4.5: Constraint suggestions (optional)
    use_suggestions = prompt_bool(
        "Would you like suggested constraint values based on your dataset?",
        default=True,
    )

    if use_suggestions:
        print("\nAnalyzing dataset characteristics...")
        advisor = ConstraintAdvisor()
        suggestions = advisor.suggest_constraints(
            dataframe=dataframe,
            task_type=task_type,
            model_family=model_family,
        )

        # Display suggestions
        print("\n" + "=" * 60)
        print("Suggested Constraint Values")
        print("=" * 60)
        print(f"Maximum features: {suggestions['max_features']}")
        print(f"Maximum interactions: {suggestions['max_interactions']}")
        print(f"Maximum cardinality: {suggestions['max_cardinality']}")
        if "reasoning" in suggestions:
            print(f"\nReasoning: {suggestions['reasoning']}")
        print("=" * 60)

        # Ask user to accept or modify
        accept_suggestions = prompt_bool("Use these suggested values?", default=True)

        if accept_suggestions:
            max_features = suggestions["max_features"]
            max_interactions = suggestions["max_interactions"]
            max_cardinality = suggestions["max_cardinality"]
            print("✓ Using suggested values")
        else:
            # Fall through to manual input
            max_features = prompt_int(
                "Maximum features",
                min_val=1,
                max_val=10000,
                default=suggestions["max_features"],
            )

            max_interactions = prompt_int(
                "Maximum feature interactions",
                min_val=0,
                max_val=1000,
                default=suggestions["max_interactions"],
            )

            max_cardinality = prompt_int(
                "Maximum cardinality for categorical features",
                min_val=2,
                max_val=1000000,
                default=suggestions["max_cardinality"],
            )
    else:
        # Original manual input flow
        max_features = prompt_int(
            "Maximum features",
            min_val=1,
            max_val=10000,
            default=100,
        )

        max_interactions = prompt_int(
            "Maximum feature interactions",
            min_val=0,
            max_val=1000,
            default=50,
        )

        max_cardinality = prompt_int(
            "Maximum cardinality for categorical features",
            min_val=2,
            max_val=1000000,
            default=10000,
        )

    # Step 5: Outlier policy
    outlier_policy = prompt_enum(
        "Select outlier policy:",
        [
            (OutlierPolicy.PRESERVE.value, "Preserve outliers"),
            (OutlierPolicy.CLIP.value, "Clip outliers to bounds"),
            (OutlierPolicy.FLAG.value, "Flag outliers with indicator features"),
        ],
        default=OutlierPolicy.PRESERVE.value,
    )

    # Step 6: Allow column dropping
    allow_column_dropping = prompt_bool(
        "Allow column dropping?",
        default=False,
    )

    # Step 7: Interpretability priority
    interpretability_priority = prompt_enum(
        "Select interpretability priority:",
        [
            (InterpretabilityPriority.LOW.value, "Low (complex features OK)"),
            (InterpretabilityPriority.MEDIUM.value, "Medium (balanced)"),
            (InterpretabilityPriority.HIGH.value, "High (simple, interpretable features only)"),
        ],
        default=InterpretabilityPriority.MEDIUM.value,
    )

    # Build intent dictionary
    intent_dict = {
        "dataset": {"path": dataset_path},
        "task": {
            "type": task_type,
            "target_column": target_column,
        },
        "model": {"family": model_family},
        "preferences": {
            "outlier_policy": outlier_policy,
            "allow_column_dropping": allow_column_dropping,
            "interpretability_priority": interpretability_priority,
        },
        "constraints": {
            "max_features": max_features,
            "max_interactions": max_interactions,
            "max_cardinality": max_cardinality,
        },
    }

    # Display summary
    display_summary(intent_dict)

    # Confirmation
    proceed = prompt_bool("Proceed with these settings?", default=True)
    if not proceed:
        print("\nCancelled by user.")
        sys.exit(1)

    return intent_dict
