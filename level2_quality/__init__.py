"""Level 2: Data Quality Agent.

This module handles dataset quality profiling, LLM-based action proposals,
and safe execution of data cleaning actions.
"""

from .agent import AgentError, DataQualityAgent
from .executor import ActionExecutor, AppliedAction, ExecutionError
from .profiler import (
    ColumnQualityProfile,
    DatasetQualityProfile,
    NumericStats,
    profile_dataset,
)

__all__ = [
    "profile_dataset",
    "DatasetQualityProfile",
    "ColumnQualityProfile",
    "NumericStats",
    "DataQualityAgent",
    "AgentError",
    "ActionExecutor",
    "AppliedAction",
    "ExecutionError",
]
