"""Core orchestration components for AgentPrep."""

__all__ = ["PipelineOrchestrator"]


def __getattr__(name):
    # Lazy: core.recipe is imported by Levels 2 and 4, which the orchestrator imports
    if name == "PipelineOrchestrator":
        from .orchestrator import PipelineOrchestrator

        return PipelineOrchestrator
    raise AttributeError(name)
