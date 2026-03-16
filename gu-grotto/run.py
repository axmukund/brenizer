#!/usr/bin/env python3
"""
Gu Refining Grotto — CLI Entry Point

Usage:
  python run.py "your request here"
  python run.py --benchmark          # Run synthetic benchmarks
  python run.py --config config.json "request"

Environment variables:
  GROTTO_API_KEY       — API key for the small model
  GROTTO_API_BASE      — API base URL (default: https://api.openai.com/v1)
  GROTTO_MODEL         — Model name (default: raptor-mini)
  GROTTO_MAX_WORKERS   — Max parallel workers (default: 12)
  GROTTO_VERBOSE       — Enable verbose logging (default: 1)
  GROTTO_FRONTIER_MODEL    — Frontier model for benchmarking
  GROTTO_FRONTIER_API_KEY  — API key for frontier model
"""

import argparse
import logging
import sys

from grotto.config import GrottoConfig
from grotto.orchestrator import GrottoOrchestrator


def setup_logging(verbose: bool = True):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Gu Refining Grotto — Multi-Agent Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("request", nargs="?", help="The request to process")
    parser.add_argument("--config", help="Path to config JSON file")
    parser.add_argument("--benchmark", action="store_true", help="Run synthetic benchmarks")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--max-workers", type=int, help="Override max workers")

    args = parser.parse_args()

    # Load config
    if args.config:
        config = GrottoConfig.from_file(args.config)
    else:
        config = GrottoConfig.from_env()

    if args.model:
        config.model.model = args.model
    if args.max_workers:
        config.orchestrator.max_workers = args.max_workers
    if args.quiet:
        config.orchestrator.verbose = False

    setup_logging(config.orchestrator.verbose)

    if args.benchmark:
        from benchmarks.runner import run_benchmarks
        run_benchmarks(config)
        return

    if not args.request:
        parser.print_help()
        sys.exit(1)

    # Execute request
    grotto = GrottoOrchestrator(config)
    try:
        report = grotto.execute(args.request)
        # Write final output to stdout
        print(report.final_output)

        if config.orchestrator.verbose:
            print("\n" + "─" * 60, file=sys.stderr)
            print(f"Complexity: {report.complexity.value}", file=sys.stderr)
            print(f"LLM calls:  {report.total_llm_calls}", file=sys.stderr)
            print(f"Tokens:     {report.total_tokens}", file=sys.stderr)
            print(f"Time:       {report.total_time_seconds:.1f}s", file=sys.stderr)
            print(f"Agents:     {', '.join(report.agents_used)}", file=sys.stderr)
            print(f"Iterations: {report.iterations}", file=sys.stderr)
    finally:
        grotto.close()


if __name__ == "__main__":
    main()
