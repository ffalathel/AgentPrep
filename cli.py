"""Command-line interface for AgentPrep.

This module provides the CLI entry point for the AgentPrep tool.
It handles argument parsing, intent validation, and orchestrator execution.
"""

import argparse
import sys
from pathlib import Path

from intent.validator import IntentValidationError, validate_intent
from core.orchestrator import PipelineOrchestrator
from utils import (
    EXIT_INVALID_INTENT,
    EXIT_POLICY_VIOLATION,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    get_llm_client,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="AgentPrep - ML preprocessing tool with AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute", required=True
    )

    # 'run' command
    run_parser = subparsers.add_parser(
        "run", help="Run the preprocessing pipeline (interactive mode)"
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
    """Run the preprocessing pipeline with a validated intent.

    Args:
        intent: Validated IntentSchema instance
        output_path: Optional path for pipeline outputs

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Get LLM client from environment variables
        llm_client = get_llm_client()
        if llm_client is None:
            print(
                "ℹ No LLM API key found. Agents will run in stub mode (no LLM proposals).\n"
                "   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable LLM features."
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
            print(f"✗ Pipeline failed with exit code: {exit_code}")

        return exit_code

    except KeyboardInterrupt:
        print("\n✗ Pipeline interrupted by user", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        print(f"✗ Runtime error: {e}", file=sys.stderr)
        from utils import get_logger
        logger = get_logger(__name__)
        logger.exception("Unexpected error during pipeline execution")
        return EXIT_RUNTIME_ERROR


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Handle commands
    if args.command == "run":
        try:
            # Collect intent interactively
            from cli.interactive import collect_intent_interactively, prompt_output_path

            intent_dict = collect_intent_interactively()

            # Validate intent
            intent = validate_intent(intent_dict)
            print("✓ Intent validated successfully")

            # Get output path (use CLI arg if provided, otherwise prompt)
            output_path = args.output
            if output_path is None:
                from cli.interactive import prompt_output_path
                output_path = prompt_output_path()

            # Run pipeline
            return run_pipeline_with_intent(intent, output_path=output_path)

        except IntentValidationError as e:
            print(f"✗ Invalid intent configuration:\n{e}", file=sys.stderr)
            return EXIT_INVALID_INTENT
        except SystemExit as e:
            # User cancelled during interactive prompts
            return e.code if e.code else EXIT_RUNTIME_ERROR
        except KeyboardInterrupt:
            print("\n\n✗ Pipeline interrupted by user", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
