"""Command-line interface for AgentPrep.

This module provides the CLI entry point for the AgentPrep tool.
It handles argument parsing, intent validation, and orchestrator execution.
"""

import argparse
import logging
import sys
from pathlib import Path

from intent.validator import IntentValidationError, load_and_validate_intent
from orchestrator import PipelineOrchestrator

# Exit codes
EXIT_SUCCESS = 0
EXIT_INVALID_INTENT = 1
EXIT_RUNTIME_ERROR = 3


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
        "run", help="Run the preprocessing pipeline"
    )
    run_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to intent configuration file (YAML or JSON)",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for pipeline outputs (optional)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def run_pipeline(config_path: str, output_path: str | None = None) -> int:
    """Run the preprocessing pipeline.

    Args:
        config_path: Path to intent configuration file
        output_path: Optional path for pipeline outputs

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Load and validate intent
        print(f"Loading configuration from: {config_path}")
        intent = load_and_validate_intent(config_path)
        print("✓ Intent validated successfully")

        # Create orchestrator
        output = Path(output_path) if output_path else None
        orchestrator = PipelineOrchestrator(intent, output_path=output)

        # Run pipeline
        print("Starting preprocessing pipeline...")
        exit_code = orchestrator.run()

        if exit_code == 0:
            print("✓ Pipeline completed successfully")
        else:
            print(f"✗ Pipeline failed with exit code: {exit_code}")

        return exit_code

    except IntentValidationError as e:
        print(f"✗ Invalid intent configuration:\n{e}", file=sys.stderr)
        return EXIT_INVALID_INTENT
    except KeyboardInterrupt:
        print("\n✗ Pipeline interrupted by user", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except Exception as e:
        print(f"✗ Runtime error: {e}", file=sys.stderr)
        logging.exception("Unexpected error during pipeline execution")
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
        return run_pipeline(
            config_path=args.config,
            output_path=args.output,
        )
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())
