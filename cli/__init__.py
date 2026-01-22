"""CLI package for AgentPrep.

This package intentionally provides a *module-like* surface area for backwards
compatibility with older tests/entrypoints that do:

    import cli
    python -m cli ...

The actual interactive wizard lives in `cli/interactive.py`, but we re-export
the common CLI symbols here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.orchestrator import PipelineOrchestrator
from intent.validator import (
    IntentValidationError,
    load_and_validate_intent,
    validate_intent,
)
from utils import (
    EXIT_INVALID_INTENT,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    setup_logging,
)

from .interactive import collect_intent_interactively

__all__ = [
    # Exit codes / logging
    "EXIT_INVALID_INTENT",
    "EXIT_RUNTIME_ERROR",
    "EXIT_SUCCESS",
    "setup_logging",
    # Intent validation
    "IntentValidationError",
    "validate_intent",
    "load_and_validate_intent",
    # Pipeline runners
    "run_pipeline",
    "run_pipeline_with_intent",
    # CLI entrypoints
    "parse_args",
    "main",
    # Interactive
    "collect_intent_interactively",
    # Re-export for unit-test patching
    "PipelineOrchestrator",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Supports:
    - Interactive run (no --config)
    - Config-file run (--config path.yaml|json)
    """
    parser = argparse.ArgumentParser(
        description="AgentPrep - ML preprocessing tool with AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the preprocessing pipeline")
    run_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON intent config (optional; if omitted, runs interactively)",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for pipeline outputs (optional, can also be set interactively)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def run_pipeline_with_intent(intent, output_path: str | None = None) -> int:
    """Run the preprocessing pipeline with a validated intent."""
    try:
        # Determine LLM provider (if any)
        try:
            from .interactive import get_selected_llm_provider, prompt_llm_provider
            from utils import get_llm_client
        except ImportError:
            get_selected_llm_provider = None  # type: ignore[assignment]
            prompt_llm_provider = None  # type: ignore[assignment]
            get_llm_client = None  # type: ignore[assignment]

        provider = None
        if "model" in intent.__dict__:
            # If in the future provider is encoded in intent, it can be read here.
            provider = None  # placeholder to keep behavior backwards-compatible

        if provider is None and get_selected_llm_provider is not None:
            provider = get_selected_llm_provider()

        # Don't prompt in non-interactive contexts (e.g., tests, scripts)
        # Only prompt if we're in an interactive terminal session
        if provider is None and prompt_llm_provider is not None:
            if sys.stdin.isatty() and sys.stdout.isatty():
                # Fallback: if not already chosen (e.g. non-interactive use), ask now
                provider = prompt_llm_provider()

        llm_client = None
        if provider is not None and get_llm_client is not None:
            llm_client = get_llm_client(preferred_provider=provider)
            if llm_client is None:
                print(
                    f"✗ Failed to initialize LLM client for provider '{provider}'.\n"
                    "  Check that the corresponding API key is set and SDK is installed.\n"
                    "  Continuing in stub mode (no LLM proposals).",
                    file=sys.stderr,
                )
        elif provider is None:
            print(
                "ℹ No LLM provider selected. Agents will run in stub mode (no LLM proposals).",
                file=sys.stderr,
            )

        # Create orchestrator
        output = Path(output_path) if output_path else None
        orchestrator = PipelineOrchestrator(
            intent, output_path=output, llm_client=llm_client
        )

        # Run pipeline
        print("\nStarting preprocessing pipeline...")
        exit_code = orchestrator.run()

        if exit_code == 0:
            print("✓ Pipeline completed successfully")
        else:
            print(f"✗ Pipeline failed with exit code: {exit_code}", file=sys.stderr)

        return exit_code

    except KeyboardInterrupt:
        print("\n✗ Pipeline interrupted by user", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except (OSError, IOError) as e:
        print(f"✗ I/O error: {e}", file=sys.stderr)
        from utils import get_logger

        logger = get_logger(__name__)
        logger.exception("I/O error during pipeline execution")
        return EXIT_RUNTIME_ERROR
    except MemoryError as e:
        print(f"✗ Insufficient memory: {e}", file=sys.stderr)
        from utils import get_logger

        logger = get_logger(__name__)
        logger.exception("Memory error during pipeline execution")
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        print(f"✗ Runtime error: {type(e).__name__}: {e}", file=sys.stderr)
        from utils import get_logger

        logger = get_logger(__name__)
        logger.exception(f"Unexpected error during pipeline execution: {type(e).__name__}")
        return EXIT_RUNTIME_ERROR


def run_pipeline(config_path: str, output_path: str | None = None) -> int:
    """Run pipeline from a YAML/JSON config file path (legacy mode)."""
    try:
        intent = load_and_validate_intent(config_path)
        return run_pipeline_with_intent(intent, output_path=output_path)
    except IntentValidationError as e:
        print(f"✗ Invalid intent configuration:\n{e}", file=sys.stderr)
        return EXIT_INVALID_INTENT
    except Exception as e:
        print(f"✗ Runtime error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


def main() -> int:
    """Main entry point for `python -m cli`."""
    args = parse_args()
    setup_logging(verbose=getattr(args, "verbose", False))

    if args.command != "run":
        return EXIT_RUNTIME_ERROR

    # Config mode
    if getattr(args, "config", None):
        return run_pipeline(args.config, output_path=getattr(args, "output", None))

    # Interactive mode
    try:
        from .interactive import collect_intent_interactively, prompt_output_path

        intent_dict = collect_intent_interactively()
        intent = validate_intent(intent_dict)
        print("✓ Intent validated successfully")

        output_path = getattr(args, "output", None)
        if output_path is None:
            output_path = prompt_output_path()

        return run_pipeline_with_intent(intent, output_path=output_path)
    except IntentValidationError as e:
        print(f"✗ Invalid intent configuration:\n{e}", file=sys.stderr)
        return EXIT_INVALID_INTENT
    except SystemExit as e:
        return int(e.code) if e.code else EXIT_RUNTIME_ERROR
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except (OSError, IOError) as e:
        print(f"✗ I/O error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
