"""
Benchmark runner for the Gu Refining Grotto.

Runs each synthetic task through:
1. The orchestrated small-model pipeline (Grotto)
2. A direct frontier model call (baseline)

Then evaluates both outputs using the meta-evaluator.
"""

from __future__ import annotations

import json
import logging
import os
import time

from grotto.config import GrottoConfig
from grotto.orchestrator import GrottoOrchestrator
from grotto import prompts
from .tasks import BENCHMARK_TASKS

logger = logging.getLogger("grotto.benchmark")


def run_benchmarks(config: GrottoConfig, task_ids: list[str] | None = None):
    """Run benchmark suite and produce comparison report."""
    grotto = GrottoOrchestrator(config)
    results = []

    tasks = BENCHMARK_TASKS
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]

    logger.info("═" * 60)
    logger.info("  GU REFINING GROTTO — BENCHMARK SUITE")
    logger.info("  Tasks: %d", len(tasks))
    logger.info("  Small model: %s", config.model.model)
    logger.info("  Frontier model: %s", config.model.frontier_model)
    logger.info("═" * 60)

    for i, task in enumerate(tasks):
        logger.info("\n" + "━" * 60)
        logger.info("Task %d/%d: %s [%s]", i + 1, len(tasks), task["name"], task["id"])
        logger.info("Request: %s", task["request"])
        logger.info("━" * 60)

        result = run_single_benchmark(grotto, task, config)
        results.append(result)

        # Print intermediate result
        logger.info("\n  Grotto score:   %.1f/50", result.get("grotto_total", 0))
        logger.info("  Frontier score: %.1f/50", result.get("frontier_total", 0))
        logger.info("  Winner: %s", result.get("winner", "?"))

    # Summary
    print_summary(results, config)

    # Save results
    save_results(results, config)

    grotto.close()


def run_single_benchmark(
    grotto: GrottoOrchestrator,
    task: dict,
    config: GrottoConfig,
) -> dict:
    """Run a single benchmark task and evaluate."""
    request = task["request"]

    # Run through Grotto (orchestrated small model)
    logger.info("\n  Running Grotto pipeline...")
    grotto_start = time.monotonic()
    grotto_report = grotto.execute(request)
    grotto_time = time.monotonic() - grotto_start
    grotto_output = grotto_report.final_output

    # Run through frontier model directly
    logger.info("\n  Running frontier model directly...")
    frontier_start = time.monotonic()
    frontier_output = grotto.execute_direct(request)
    frontier_time = time.monotonic() - frontier_start

    # Evaluate both outputs
    logger.info("\n  Evaluating outputs...")
    evaluation = evaluate_outputs(grotto, request, grotto_output, frontier_output, config)

    return {
        "task_id": task["id"],
        "task_name": task["name"],
        "request": request,
        "expected_complexity": task["expected_complexity"],
        "actual_complexity": grotto_report.complexity.value,
        "grotto_output_length": len(grotto_output),
        "frontier_output_length": len(frontier_output),
        "grotto_time": grotto_time,
        "frontier_time": frontier_time,
        "grotto_llm_calls": grotto_report.total_llm_calls,
        "grotto_tokens": grotto_report.total_tokens,
        "grotto_agents": grotto_report.agents_used,
        "grotto_total": evaluation.get("output_a_total", 0),
        "frontier_total": evaluation.get("output_b_total", 0),
        "winner": evaluation.get("winner", "?"),
        "analysis": evaluation.get("analysis", ""),
        "evaluation": evaluation,
    }


def evaluate_outputs(
    grotto: GrottoOrchestrator,
    request: str,
    grotto_output: str,
    frontier_output: str,
    config: GrottoConfig,
) -> dict:
    """Use the meta-evaluator to compare outputs."""
    prompt = prompts.META_EVALUATOR_COMPARE.format(
        request=request,
        output_a=grotto_output[:4000],  # Truncate to fit context
        output_b=frontier_output[:4000],
    )

    # Use frontier model as the judge for fairness
    result = grotto.llm.chat_json(
        messages=[
            {"role": "system", "content": prompts.META_EVALUATOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        model=config.model.frontier_model,
        temperature=0.1,
    )
    return result


def print_summary(results: list[dict], config: GrottoConfig):
    """Print a summary table of all benchmark results."""
    print("\n" + "═" * 80)
    print("  BENCHMARK SUMMARY")
    print("═" * 80)
    print(f"  Small model:    {config.model.model}")
    print(f"  Frontier model: {config.model.frontier_model}")
    print("─" * 80)
    print(f"  {'Task':<35} {'Grotto':>8} {'Frontier':>8} {'Winner':>8} {'Calls':>6}")
    print("─" * 80)

    grotto_wins = 0
    frontier_wins = 0
    ties = 0

    for r in results:
        winner_str = r["winner"]
        if r["winner"] == "A":
            winner_str = "GROTTO"
            grotto_wins += 1
        elif r["winner"] == "B":
            winner_str = "FRNTR"
            frontier_wins += 1
        else:
            winner_str = "TIE"
            ties += 1

        print(f"  {r['task_name']:<35} {r['grotto_total']:>8.1f} {r['frontier_total']:>8.1f} "
              f"{winner_str:>8} {r['grotto_llm_calls']:>6}")

    print("─" * 80)
    total_grotto = sum(r["grotto_total"] for r in results)
    total_frontier = sum(r["frontier_total"] for r in results)
    print(f"  {'TOTAL':<35} {total_grotto:>8.1f} {total_frontier:>8.1f}")
    print(f"\n  Grotto wins: {grotto_wins} | Frontier wins: {frontier_wins} | Ties: {ties}")
    print(f"  Win rate: {grotto_wins/len(results)*100:.0f}%")
    print(f"  Avg Grotto score: {total_grotto/len(results):.1f}/50")
    print(f"  Avg Frontier score: {total_frontier/len(results):.1f}/50")
    print("═" * 80)


def save_results(results: list[dict], config: GrottoConfig):
    """Save benchmark results to JSON file."""
    os.makedirs("benchmark_results", exist_ok=True)
    filename = f"benchmark_results/run_{int(time.time())}.json"

    data = {
        "config": {
            "small_model": config.model.model,
            "frontier_model": config.model.frontier_model,
            "max_workers": config.orchestrator.max_workers,
            "quality_threshold": config.orchestrator.quality_threshold,
            "ensemble_voting": config.orchestrator.ensemble_voting,
            "ensemble_size": config.orchestrator.ensemble_size,
            "adversarial_review": config.orchestrator.adversarial_review,
            "synthesis_iterations": config.orchestrator.synthesis_iterations,
        },
        "results": results,
        "summary": {
            "total_tasks": len(results),
            "grotto_wins": sum(1 for r in results if r["winner"] == "A"),
            "frontier_wins": sum(1 for r in results if r["winner"] == "B"),
            "ties": sum(1 for r in results if r["winner"] == "TIE"),
            "avg_grotto_score": sum(r["grotto_total"] for r in results) / max(len(results), 1),
            "avg_frontier_score": sum(r["frontier_total"] for r in results) / max(len(results), 1),
        },
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", filename)
