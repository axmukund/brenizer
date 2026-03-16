"""
Fang Yuan (方源) — The Grand Orchestrator

Main entry point for the Gu Refining Grotto. Receives requests,
classifies complexity, dispatches to agents, manages the full
lifecycle of a "convoy" (end-to-end task execution).

Design iteration v4 — key improvements over v1:
- Adaptive worker count based on complexity classification
- Dream Realm context sharing between phases
- Spring Autumn Cicada handoffs for long-running tasks
- Ensemble voting for quality gates
- Progressive synthesis with iterative refinement
"""

from __future__ import annotations

import json
import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .config import GrottoConfig
from .llm import LLMClient
from .types import (
    Aperture,
    Complexity,
    ConvoyReport,
    DaoMark,
    DreamShard,
    GuWorm,
    GuWormStatus,
    KillerMove,
)
from . import prompts

logger = logging.getLogger("grotto.orchestrator")

# v2 fix: estimate tokens from character count (rough heuristic)
_CHARS_PER_TOKEN = 4

def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


class GrottoOrchestrator:
    """
    The Grand Orchestrator — Fang Yuan.

    Manages the complete lifecycle:
    1. Classify request complexity
    2. Decompose into subtasks (Star Constellation)
    3. Dispatch workers (Giant Sun)
    4. Execute subtasks in parallel (Rank 5 Gu Masters)
    5. Review outputs (Spectral Soul)
    6. Synthesize final output (Thieving Heaven)
    7. Track completion (Paradise Earth)
    """

    def __init__(self, config: GrottoConfig | None = None):
        self.config = config or GrottoConfig.from_env()
        self.llm = LLMClient(self.config.model)
        self.aperture = Aperture(name="default")
        self._log_header()

    def _log_header(self):
        logger.info("═" * 60)
        logger.info("  Gu Refining Grotto v0.4.0")
        logger.info("  Model: %s", self.config.model.model)
        logger.info("  Max Workers: %d", self.config.orchestrator.max_workers)
        logger.info("═" * 60)

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def execute(self, request: str) -> ConvoyReport:
        """
        Execute a request end-to-end. This is the main entry point.
        Spawns subagents with great abandon.
        """
        start_time = time.monotonic()
        report = ConvoyReport(request=request)
        # v2: store original request for full-context propagation
        self._original_request = request

        # Phase 1: Classify complexity
        logger.info("\n🔮 Phase 1: Fang Yuan classifies the request...")
        assessment = self._classify_request(request)
        # v2 fix: only treat as trivial if request is very short
        raw_complexity = assessment.get("complexity", "moderate")
        if raw_complexity == "trivial" and len(request.split()) > 10:
            raw_complexity = "simple"  # don't gate longer requests as trivial
        complexity = Complexity(raw_complexity)
        report.complexity = complexity
        logger.info("   Complexity: %s | Subtasks: ~%d",
                     complexity.value, assessment.get("estimated_subtasks", 0))

        # Trivial requests: answer directly (only for very short requests)
        if complexity == Complexity.TRIVIAL and assessment.get("direct_answer"):
            report.final_output = assessment["direct_answer"]
            report.total_time_seconds = time.monotonic() - start_time
            report.total_llm_calls = self.llm.total_calls
            report.total_tokens = self.llm.total_tokens
            report.agents_used = ["fang_yuan"]
            logger.info("   Direct answer (trivial)")
            return report

        # Phase 2: Decompose into subtasks
        # v4: Myriad Self tournament for complex/profound tasks
        if (complexity in (Complexity.COMPLEX, Complexity.PROFOUND)
                and self.config.orchestrator.myriad_self_enabled):
            logger.info("\n📐 Phase 2: Myriad Self tournament decomposition...")
            plan = self._myriad_self_decompose(request, assessment)
        else:
            logger.info("\n📐 Phase 2: Star Constellation decomposes the task...")
            plan = self._decompose_task(request, assessment)

        # v2 fix: check decomposability — route to single-worker path if needed
        decomposable = plan.get("decomposable", True)
        total_subtasks = plan.get("total_subtasks", 0)
        if not decomposable or total_subtasks <= 1:
            logger.info("   Task is indivisible — routing to single-worker path")
            return self._single_worker_path(request, assessment, start_time)

        killer_move = self._plan_to_killer_move(plan)
        report.killer_move = killer_move
        logger.info("   Killer Move: %s | %d subtasks in %d phases",
                     plan.get("killer_move_name", "unnamed"),
                     len(killer_move.gu_worms),
                     len(plan.get("phases", [])))

        # v2: generate shared assumptions before parallel dispatch
        shared_assumptions = self._generate_shared_assumptions(request, plan)

        # Phase 3: Execute phase by phase
        logger.info("\n⚔️  Phase 3: Executing Killer Move...")
        phase_outputs = self._execute_killer_move(killer_move, plan, shared_assumptions)

        # Phase 4: Synthesize (v2: hierarchical if needed)
        logger.info("\n🌀 Phase 4: Thieving Heaven synthesizes outputs...")
        final_output = self._synthesize_outputs(request, plan, phase_outputs)

        # Phase 5: Final review
        logger.info("\n👁️  Phase 5: Spectral Soul final review...")
        final_output = self._final_review_loop(request, final_output)

        # Wrap up
        report.final_output = final_output
        report.total_time_seconds = time.monotonic() - start_time
        report.total_llm_calls = self.llm.total_calls
        report.total_tokens = self.llm.total_tokens
        report.agents_used = self._collect_agent_names(killer_move)
        report.iterations = max(w.attempt for w in killer_move.gu_worms) if killer_move.gu_worms else 1

        logger.info("\n✅ Convoy complete! %.1fs | %d LLM calls | %d tokens",
                     report.total_time_seconds, report.total_llm_calls, report.total_tokens)

        if self.config.orchestrator.persist_state:
            self._persist_report(report)

        return report

    def _single_worker_path(self, request: str, assessment: dict, start_time: float) -> ConvoyReport:
        """
        v2: For indivisible tasks, use a single worker with extended
        chain-of-thought and multiple critic cycles instead of decomposition.
        """
        report = ConvoyReport(request=request, complexity=Complexity.SIMPLE)
        worm = GuWorm(
            title="Complete task (single worker)",
            description=(
                f"Original request: {request}\n\n"
                f"Assessment: {json.dumps(assessment, indent=2)}\n\n"
                "Think step by step. This task should NOT be decomposed — "
                "address all aspects in a single, coherent response."
            ),
            metadata={"output_format": "text"},
        )
        output = self._execute_single_worm(worm, [])
        report.final_output = worm.output
        report.total_time_seconds = time.monotonic() - start_time
        report.total_llm_calls = self.llm.total_calls
        report.total_tokens = self.llm.total_tokens
        report.agents_used = ["fang_yuan", "single_worker", "spectral_soul"]
        return report

    def execute_direct(self, request: str, model: str | None = None) -> str:
        """Execute a request directly with a single LLM call (for benchmarking)."""
        resp = self.llm.chat(
            messages=[
                {"role": "system", "content": "You are a helpful expert assistant. Be thorough and specific."},
                {"role": "user", "content": request},
            ],
            model=model or self.config.model.frontier_model,
            temperature=0.3,
            max_tokens=self.config.model.max_tokens,
        )
        return resp.content

    # ──────────────────────────────────────────────────────────────
    # PHASE 1: CLASSIFICATION (Fang Yuan)
    # ──────────────────────────────────────────────────────────────

    def _classify_request(self, request: str) -> dict:
        """Classify request complexity using Fang Yuan prompt."""
        result = self.llm.chat_json(
            messages=[
                {"role": "system", "content": prompts.FANG_YUAN_SYSTEM},
                {"role": "user", "content": request},
            ],
            temperature=0.1,
        )
        return result

    # ──────────────────────────────────────────────────────────────
    # PHASE 2: DECOMPOSITION (Star Constellation)
    # ──────────────────────────────────────────────────────────────

    def _decompose_task(self, request: str, assessment: dict) -> dict:
        """Decompose task into subtasks using Star Constellation.
        v3: Uses Wisdom Gu (frontier model) for complex/profound tasks."""
        context = self._gather_dream_context(request)

        prompt = prompts.STAR_CONSTELLATION_DECOMPOSE.format(
            request=request,
            assessment=json.dumps(assessment, indent=2),
            context=context or "(no prior context)",
        )

        # v3: Wisdom Gu — frontier model for decomposition on complex tasks
        model = None
        complexity = assessment.get("complexity", "moderate")
        if (self.config.orchestrator.wisdom_gu_enabled
                and complexity in ("complex", "profound")
                and self.config.model.frontier_api_key):
            model = self.config.model.frontier_model
            logger.info("   \U0001f52e Wisdom Gu activated for decomposition (complexity: %s)", complexity)

        result = self.llm.chat_json(
            messages=[
                {"role": "system", "content": prompts.STAR_CONSTELLATION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            model=model,
        )
        return result

    def _plan_to_killer_move(self, plan: dict) -> KillerMove:
        """Convert a Star Constellation plan into a KillerMove with GuWorms."""
        km = KillerMove(
            name=plan.get("killer_move_name", "unnamed"),
            description=json.dumps(plan, indent=2),
        )

        # Flatten all subtasks from all phases
        phase_order = []
        for phase_data in plan.get("phases", []):
            phase_ids = []
            for subtask in phase_data.get("subtasks", []):
                worm = GuWorm(
                    id=subtask.get("id", f"gu-{len(km.gu_worms)}"),
                    title=subtask.get("title", "untitled"),
                    description=subtask.get("description", ""),
                    dependencies=subtask.get("dependencies", []),
                    metadata={
                        "output_format": subtask.get("output_format", "text"),
                        "difficulty": subtask.get("estimated_difficulty", "medium"),
                        "context_needed": subtask.get("context_needed", ""),
                        "phase": phase_data.get("phase", 0),
                    },
                )
                km.gu_worms.append(worm)
                phase_ids.append(worm.id)
            phase_order.append(phase_ids)

        km.execution_order = phase_order
        return km

    # ──────────────────────────────────────────────────────────────
    # MYRIAD SELF (万我) — Tournament Decomposition
    # v4: Explore multiple decomposition strategies, probe phase 1
    # of each, select the one that produces the best results.
    # ──────────────────────────────────────────────────────────────

    def _myriad_self_decompose(self, request: str, assessment: dict) -> dict:
        """
        v4 game-changer: Generate N independent decomposition plans at
        different temperatures, execute only phase 1 of each in parallel,
        score the probe outputs, and commit to the winning plan.

        This is beam search at the planning level — something a single
        frontier model fundamentally cannot do.
        """
        n_candidates = self.config.orchestrator.myriad_self_candidates
        temps = [0.1 + 0.25 * i for i in range(n_candidates)]  # e.g. [0.1, 0.35, 0.6]
        logger.info("   Myriad Self: generating %d candidate decompositions", n_candidates)

        # Step 1: Generate N decomposition plans in parallel
        context = self._gather_dream_context(request)
        prompt = prompts.STAR_CONSTELLATION_DECOMPOSE.format(
            request=request,
            assessment=json.dumps(assessment, indent=2),
            context=context or "(no prior context)",
        )

        candidates: list[dict] = []
        with ThreadPoolExecutor(max_workers=n_candidates) as pool:
            futures = []
            for temp in temps:
                future = pool.submit(
                    self.llm.chat_json,
                    messages=[
                        {"role": "system", "content": prompts.STAR_CONSTELLATION_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                )
                futures.append(future)
            for future in as_completed(futures):
                try:
                    plan = future.result()
                    if plan.get("phases"):
                        candidates.append(plan)
                except Exception as e:
                    logger.warning("   Myriad Self candidate failed: %s", e)

        if len(candidates) <= 1:
            logger.info("   Only %d viable candidate — using it directly", len(candidates))
            return candidates[0] if candidates else self._decompose_task(request, assessment)

        # Step 2: Probe — execute phase 1 of each candidate
        logger.info("   Probing phase 1 of %d candidates...", len(candidates))
        probes: list[tuple[dict, float]] = []

        for plan in candidates:
            km = self._plan_to_killer_move(plan)
            if not km.execution_order:
                probes.append((plan, 0.0))
                continue

            phase1_ids = km.execution_order[0]
            phase1_worms = [w for w in km.gu_worms if w.id in phase1_ids]
            if not phase1_worms:
                probes.append((plan, 0.0))
                continue

            # Execute phase 1 workers (no review loop — just raw worker output)
            phase1_results = self._execute_worms_parallel(phase1_worms, [])

            # Score probe outputs via quick critic pass
            total_score = 0.0
            scored = 0
            for worm in phase1_worms:
                if worm.output:
                    review = self._run_critic(worm)
                    total_score += review.get("overall_score", 0.0)
                    scored += 1

            avg_score = total_score / max(scored, 1)
            probes.append((plan, avg_score))
            plan_name = plan.get("killer_move_name", "unnamed")
            logger.info("   Candidate '%s': probe score %.2f (%d subtasks)",
                        plan_name[:30], avg_score, plan.get("total_subtasks", 0))

        # Step 3: Select the winner
        winner, best_score = max(probes, key=lambda p: p[1])
        logger.info("   🏆 Myriad Self winner: '%s' (score: %.2f)",
                     winner.get("killer_move_name", "unnamed")[:30], best_score)
        return winner

    # ──────────────────────────────────────────────────────────────
    # PHASE 3: EXECUTION (Giant Sun dispatch + Rank 5 workers)
    # ──────────────────────────────────────────────────────────────

    def _generate_shared_assumptions(self, request: str, plan: dict) -> str:
        """
        v2: Generate shared assumptions that all parallel workers receive.
        Prevents architectural contradictions between workers.
        """
        subtask_titles = []
        for phase in plan.get("phases", []):
            for st in phase.get("subtasks", []):
                subtask_titles.append(st.get("title", "unnamed"))

        resp = self.llm.chat(
            messages=[
                {"role": "system", "content": "You produce a brief shared-assumptions preamble. "
                 "Given a task and its subtasks, identify architectural choices, technology "
                 "assumptions, and design constraints that ALL workers must agree on. "
                 "Be concise: bullet points, max 300 words."},
                {"role": "user", "content": f"Request: {request}\n\n"
                 f"Subtasks: {', '.join(subtask_titles)}\n\n"
                 "List the shared assumptions all workers must follow:"},
            ],
            temperature=0.2,
        )
        logger.info("   Shared assumptions generated (%d chars)", len(resp.content))
        return resp.content

    def _execute_killer_move(self, km: KillerMove, plan: dict, shared_assumptions: str = "") -> dict[str, str]:
        """Execute all phases of a KillerMove, collecting outputs."""
        all_outputs: dict[str, str] = {}
        phase_context: list[str] = []
        if shared_assumptions:
            phase_context.append(f"[SHARED ASSUMPTIONS]:\n{shared_assumptions}")

        for phase_idx, phase_ids in enumerate(km.execution_order):
            phase_worms = [w for w in km.gu_worms if w.id in phase_ids]
            if not phase_worms:
                continue

            logger.info("   Phase %d: %d subtasks [%s]",
                         phase_idx + 1, len(phase_worms),
                         ", ".join(w.title[:30] for w in phase_worms))

            # Execute worms in parallel within each phase
            phase_results = self._execute_worms_parallel(phase_worms, phase_context)

            for worm_id, output in phase_results.items():
                all_outputs[worm_id] = output

            # v2: extractive summarization for phase context instead of truncation
            for w in phase_worms:
                if w.status in (GuWormStatus.REFINED, GuWormStatus.REVIEWED):
                    summary = self._summarize_for_context(w)
                    phase_context.append(f"[{w.title}]: {summary}")
                    # Store as Dream Shard
                    self.aperture.dream_shards.append(DreamShard(
                        content=w.output[:2000],
                        source_agent=w.assigned_to or "worker",
                        relevance_tags=[w.title],
                    ))
                elif w.status == GuWormStatus.FAILED:
                    # v2: explicitly note failures for synthesis
                    phase_context.append(f"[{w.title}]: FAILED — could not be completed")

            # v2: reconcile phase outputs if multiple parallel tasks
            if len(phase_worms) > 1 and phase_idx < len(km.execution_order) - 1:
                self._reconcile_phase(phase_worms, phase_context)

        return all_outputs

    def _summarize_for_context(self, worm: GuWorm) -> str:
        """
        v3: Python-based extractive summary for phase handoff context.
        No LLM call needed — uses keyword extraction for key lines.
        """
        if len(worm.output) <= 800:
            return worm.output
        lines = worm.output.split('\n')
        key_keywords = ['decision', 'conclusion', 'therefore', 'recommendation',
                        'interface', 'api', 'result', 'approach', 'constraint',
                        'assumption', 'important', 'must', 'require', 'output']
        key_lines = [l for l in lines
                     if any(kw in l.lower() for kw in key_keywords) and l.strip()]
        # first 3 lines + key lines + last 2 lines, deduped
        selected = lines[:3] + key_lines[:8] + lines[-2:]
        seen = set()
        deduped = []
        for line in selected:
            if line not in seen:
                seen.add(line)
                deduped.append(line)
        return '\n'.join(deduped)

    def _reconcile_phase(self, worms: list[GuWorm], phase_context: list[str]):
        """
        v2: Check for contradictions between parallel workers' outputs.
        """
        successful = [w for w in worms if w.status in (GuWormStatus.REFINED, GuWormStatus.REVIEWED)]
        if len(successful) < 2:
            return
        outputs_summary = "\n---\n".join(
            f"{w.title}: {w.output[:500]}" for w in successful
        )
        resp = self.llm.chat(
            messages=[
                {"role": "system", "content": "Check these parallel task outputs for contradictions. "
                 "If they make conflicting assumptions, list the conflicts. "
                 "If consistent, say CONSISTENT. Be brief."},
                {"role": "user", "content": outputs_summary},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        if "CONSISTENT" not in resp.content.upper():
            logger.warning("   ⚠ Phase reconciliation found conflicts: %s", resp.content[:100])
            phase_context.append(f"[RECONCILIATION WARNING]: {resp.content}")

    def _execute_worms_parallel(
        self, worms: list[GuWorm], context: list[str]
    ) -> dict[str, str]:
        """Execute a batch of GuWorms in parallel using thread pool."""
        results: dict[str, str] = {}
        max_par = min(len(worms), self.config.orchestrator.max_workers)

        with ThreadPoolExecutor(max_workers=max_par) as pool:
            futures = {}
            for worm in worms:
                worm.status = GuWormStatus.REFINING
                worm.assigned_to = f"gu_master_{worm.id}"
                future = pool.submit(self._execute_single_worm, worm, context)
                futures[future] = worm

            for future in as_completed(futures):
                worm = futures[future]
                try:
                    output = future.result()
                    results[worm.id] = output
                except Exception as e:
                    logger.error("Worker failed on %s: %s", worm.title, e)
                    worm.status = GuWormStatus.FAILED
                    results[worm.id] = f"[FAILED: {e}]"

        return results

    def _execute_single_worm(self, worm: GuWorm, context: list[str]) -> str:
        """Execute a single GuWorm with review loop."""
        revision_feedback = ""

        for attempt in range(1, worm.max_attempts + 1):
            worm.attempt = attempt

            # Worker execution
            output = self._run_worker(worm, context, revision_feedback)
            worm.output = output

            # Critic review
            if self.config.orchestrator.adversarial_review:
                review = self._run_critic(worm)
                verdict = review.get("verdict", "APPROVED")
                score = review.get("overall_score", 0.5)

                if verdict == "APPROVED" or score >= self.config.orchestrator.quality_threshold:
                    worm.status = GuWormStatus.REVIEWED
                    worm.completed_at = time.time()
                    logger.info("      ✓ %s (score: %.2f, attempt %d)", worm.title[:30], score, attempt)

                    # Record Dao Mark
                    self.aperture.dao_marks.append(DaoMark(
                        gu_worm_id=worm.id,
                        worker_id=worm.assigned_to or "",
                        output_summary=output[:200],
                        quality_score=score,
                    ))
                    return output

                # v4: backtracking — if root_cause is not "worker", stop retrying
                root_cause = review.get("root_cause", "worker")
                if (self.config.orchestrator.backtracking_enabled
                        and root_cause in ("upstream", "decomposition")):
                    logger.warning("      ↩ %s: critic blames %s (not worker) — aborting retries",
                                     worm.title[:30], root_cause)
                    worm.status = GuWormStatus.FAILED
                    worm.completed_at = time.time()
                    worm.critique = review.get("revision_instructions", "")
                    return (f"[SUBTASK FAILED — root cause: {root_cause}]\n"
                            f"Critic analysis: {review.get('revision_instructions', 'N/A')}\n"
                            f"Backtrack target: {review.get('backtrack_to', 'N/A')}\n"
                            f"Best effort output:\n{worm.output}")

                revision_feedback = review.get("revision_instructions", "")
                worm.critique = revision_feedback
                logger.info("      ✗ %s rejected (score: %.2f, attempt %d): %s",
                             worm.title[:30], score, attempt, revision_feedback[:80])
            else:
                # No review — accept directly
                worm.status = GuWormStatus.REFINED
                worm.completed_at = time.time()
                return output

        # v2 fix: max attempts exhausted — mark as FAILED, not silently accepted
        logger.warning("      ⚠ %s: max attempts exhausted, marking FAILED", worm.title[:30])
        worm.status = GuWormStatus.FAILED
        worm.completed_at = time.time()
        # Return output with failure annotation so synthesis can handle the gap
        return f"[SUBTASK FAILED after {worm.max_attempts} attempts]\n" \
               f"Best effort output:\n{worm.output}\n" \
               f"Last critic feedback: {worm.critique}"

    def _run_worker(self, worm: GuWorm, context: list[str], revision_feedback: str) -> str:
        """Run a Rank 5 Gu Master on a single task.
        v4: Injects task-type-specific reasoning scaffold from WORKER_SCAFFOLDS."""
        context_str = "\n\n".join(context[-5:]) if context else "(first phase — no prior context)"
        # v2 fix: include full original request so workers never lose global context
        original_request = getattr(self, '_original_request', '')
        prompt = prompts.WORKER_EXECUTE.format(
            title=worm.title,
            description=worm.description,
            output_format=worm.metadata.get("output_format", "text"),
            context=context_str,
            revision_feedback=revision_feedback or "(first attempt)",
        )
        if original_request:
            prompt = f"## Original User Request\n{original_request}\n\n{prompt}"

        # v4: select scaffold based on task-type keywords in the subtask
        task_text = f"{worm.title} {worm.description}".lower()
        if any(kw in task_text for kw in ("code", "implement", "function", "class", "algorithm", "program", "script")):
            scaffold_key = "code"
        elif any(kw in task_text for kw in ("analyze", "evaluate", "compare", "review", "assess", "investigate")):
            scaffold_key = "analysis"
        elif any(kw in task_text for kw in ("design", "architect", "plan", "propose", "blueprint", "schema")):
            scaffold_key = "design"
        else:
            scaffold_key = "default"
        scaffold = prompts.WORKER_SCAFFOLDS.get(scaffold_key, prompts.WORKER_SCAFFOLDS["default"])
        system_prompt = prompts.WORKER_SYSTEM + "\n" + scaffold

        resp = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.content

    def _run_critic(self, worm: GuWorm) -> dict:
        """Run Spectral Soul to review a worker's output.
        v3: Checklist-based scoring computed in Python."""
        prompt = prompts.SPECTRAL_SOUL_REVIEW.format(
            task_description=f"{worm.title}\n\n{worm.description}",
            output=worm.output,
            previous_critique=worm.critique or "(first review)",
        )

        result = self.llm.chat_json(
            messages=[
                {"role": "system", "content": prompts.SPECTRAL_SOUL_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        # v3: compute score from checklist in Python (not LLM-reported)
        checklist = result.get("checklist", {})
        if isinstance(checklist, dict) and checklist:
            # Positive checks: True = good
            positive_keys = ["addresses_task", "format_correct", "all_requirements",
                             "specific_enough", "concurrent_safety", "edge_cases"]
            # Negative checks: False = good
            negative_keys = ["factual_errors", "would_embarrass"]

            score_points = 0
            total_points = 0
            for k in positive_keys:
                if k in checklist:
                    total_points += 1
                    if checklist[k]:
                        score_points += 1
            for k in negative_keys:
                if k in checklist:
                    total_points += 1
                    if not checklist[k]:  # False on negative = good
                        score_points += 1

            result["overall_score"] = score_points / max(total_points, 1)
        else:
            # Fallback: try old-style scores
            scores = result.get("scores", {})
            if isinstance(scores, dict) and scores:
                numeric_scores = []
                for v in scores.values():
                    try:
                        numeric_scores.append(float(v))
                    except (TypeError, ValueError):
                        pass
                if numeric_scores:
                    result["overall_score"] = sum(numeric_scores) / len(numeric_scores)

        # v4 fix: guarantee overall_score exists (prevents KeyError downstream)
        result.setdefault("overall_score", 0.0)

        # v2 fix: default to REJECTED on parse failure (fail-closed)
        if "verdict" not in result or result["verdict"] not in ("APPROVED", "REJECTED", "NEEDS_REVISION"):
            result["verdict"] = "REJECTED"
            result.setdefault("revision_instructions", "Output quality could not be assessed. Revise and be more specific.")

        return result

    # ──────────────────────────────────────────────────────────────
    # PHASE 4: SYNTHESIS (Thieving Heaven)
    # ──────────────────────────────────────────────────────────────

    def _synthesize_outputs(self, request: str, plan: dict, outputs: dict[str, str]) -> str:
        """Merge all worker outputs into a coherent final result.
        v2: uses hierarchical synthesis when outputs exceed context budget."""
        # Build worker outputs block
        worker_blocks = []
        for worm_id, output in outputs.items():
            title = worm_id
            for phase in plan.get("phases", []):
                for st in phase.get("subtasks", []):
                    if st.get("id") == worm_id:
                        title = st.get("title", worm_id)
                        break
            worker_blocks.append((title, output))

        # v2: estimate total tokens and use hierarchical synthesis if needed
        total_output_tokens = sum(_estimate_tokens(o) for _, o in worker_blocks)
        context_budget = int(self.config.model.max_tokens * 0.6)

        if total_output_tokens > context_budget and len(worker_blocks) > 2:
            logger.info("   Using hierarchical synthesis (%d tokens > %d budget)",
                         total_output_tokens, context_budget)
            return self._hierarchical_synthesize(request, plan, worker_blocks)

        worker_outputs_str = "\n\n---\n\n".join(
            f"### {title}\n{output}" for title, output in worker_blocks
        )

        # Gather critic notes from Dream Realm
        critic_notes = "\n".join(
            ds.content[:300] for ds in self.aperture.dream_shards
            if "critic" in ds.source_agent.lower()
        ) or "(no critic notes)"

        prompt = prompts.THIEVING_HEAVEN_MERGE.format(
            request=request,
            plan_summary=plan.get("killer_move_name", "unnamed workflow"),
            worker_outputs=worker_outputs_str,
            critic_notes=critic_notes,
        )

        # Iterative synthesis — refine N times
        result = ""
        for i in range(self.config.orchestrator.synthesis_iterations):
            if i == 0:
                messages = [
                    {"role": "system", "content": prompts.THIEVING_HEAVEN_SYSTEM},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [
                    {"role": "system", "content": prompts.THIEVING_HEAVEN_SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": result},
                    {"role": "user", "content": f"Iteration {i+1}: Refine and improve this synthesis. "
                     "Fix any gaps, improve flow, strengthen weak sections."},
                ]

            resp = self.llm.chat(messages=messages, temperature=0.3)
            result = resp.content
            logger.info("   Synthesis iteration %d complete (%d chars)", i + 1, len(result))

        return result

    def _hierarchical_synthesize(self, request: str, plan: dict, blocks: list[tuple[str, str]]) -> str:
        """
        v2: Tree-reduction synthesis — merge outputs in pairs, then merge
        the intermediate summaries, until we get a single coherent output.
        Prevents context window overflow for large task sets.
        """
        current_level = [f"### {title}\n{output}" for title, output in blocks]

        while len(current_level) > 1:
            next_level = []
            # Process in pairs
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    pair = f"{current_level[i]}\n\n---\n\n{current_level[i+1]}"
                else:
                    next_level.append(current_level[i])
                    continue

                resp = self.llm.chat(
                    messages=[
                        {"role": "system", "content": prompts.THIEVING_HEAVEN_SYSTEM},
                        {"role": "user", "content": f"Merge these outputs for: {request}\n\n{pair}"},
                    ],
                    temperature=0.3,
                )
                next_level.append(resp.content)
                logger.info("   Hierarchical merge: %d→%d blocks remaining",
                             len(current_level), len(next_level))

            current_level = next_level

        return current_level[0] if current_level else ""

    # ──────────────────────────────────────────────────────────────
    # PHASE 5: FINAL REVIEW (Spectral Soul + Ensemble)
    # ──────────────────────────────────────────────────────────────

    def _final_review_loop(self, request: str, output: str) -> str:
        """Final quality gate with optional ensemble voting."""
        for cycle in range(self.config.orchestrator.max_review_cycles):
            if self.config.orchestrator.ensemble_voting:
                # Multiple voters assess the output
                votes = self._ensemble_vote(request, output)
                accept_count = sum(1 for v in votes if v.get("vote") == "ACCEPT")
                total = len(votes)
                logger.info("   Ensemble vote: %d/%d ACCEPT", accept_count, total)

                if accept_count > total / 2:
                    return output
            else:
                # Single critic review
                review = self._review_final_output(request, output)
                if review.get("verdict") == "APPROVED":
                    return output

            # If rejected, collect ALL rejection reasons (v4 fix)
            logger.info("   Final review cycle %d: refining...", cycle + 1)
            if self.config.orchestrator.ensemble_voting:
                all_reasons = " | ".join(
                    v.get("one_line_reason", "") for v in votes
                    if v.get("vote") == "REJECT" and v.get("one_line_reason")
                )
                feedback = all_reasons or votes[0].get("one_line_reason", "")
            else:
                feedback = review.get("revision_instructions", "")
            output = self._refine_output(request, output, feedback)

        return output  # Accept after max cycles

    def _ensemble_vote(self, request: str, output: str) -> list[dict]:
        """
        v2: Specialized ensemble voting — each voter has a different focus
        area to reduce correlated failures.
        """
        votes = []
        # v2: each voter gets a different evaluation focus
        voter_specializations = [
            "Focus on TECHNICAL ACCURACY. Is the output factually and technically correct? "
            "Look for errors in logic, incorrect claims, and wrong assumptions.",
            "Focus on COMPLETENESS. Does the output fully address the original request? "
            "Check if any aspects were missed or only superficially covered.",
            "Play DEVIL'S ADVOCATE. Your job is to find at least one serious problem "
            "with this output. If you truly cannot find any, explain why in detail.",
        ]
        # Extend or truncate to ensemble_size
        while len(voter_specializations) < self.config.orchestrator.ensemble_size:
            voter_specializations.append(voter_specializations[len(voter_specializations) % 3])
        voter_specializations = voter_specializations[:self.config.orchestrator.ensemble_size]

        prompt_base = f"Task: {request}\n\nOutput to evaluate:\n{output}"

        with ThreadPoolExecutor(max_workers=self.config.orchestrator.ensemble_size) as pool:
            futures = []
            for spec in voter_specializations:
                future = pool.submit(
                    self.llm.chat_json,
                    messages=[
                        {"role": "system", "content": prompts.ENSEMBLE_VOTER_SYSTEM + f"\n\n{spec}"},
                        {"role": "user", "content": prompt_base},
                    ],
                    temperature=0.5,
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    vote = future.result()
                    # v2: fail-closed on parse failure
                    if "vote" not in vote or vote["vote"] not in ("ACCEPT", "REJECT"):
                        vote = {"vote": "REJECT", "confidence": 0.5, "one_line_reason": "voter parse failure"}
                    votes.append(vote)
                except Exception as e:
                    logger.warning("Ensemble voter failed: %s", e)
                    # v2: fail-closed instead of fail-open
                    votes.append({"vote": "REJECT", "confidence": 0.5, "one_line_reason": f"voter error: {e}"})

        return votes

    def _review_final_output(self, request: str, output: str) -> dict:
        """Single critic review of final output."""
        prompt = prompts.SPECTRAL_SOUL_REVIEW.format(
            task_description=request,
            output=output,
            previous_critique="(final review)",
        )
        return self.llm.chat_json(
            messages=[
                {"role": "system", "content": prompts.SPECTRAL_SOUL_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

    def _refine_output(self, request: str, output: str, feedback: str) -> str:
        """Refine output based on feedback."""
        resp = self.llm.chat(
            messages=[
                {"role": "system", "content": prompts.THIEVING_HEAVEN_SYSTEM},
                {"role": "user", "content": f"Refine this output based on feedback.\n\n"
                 f"Original request: {request}\n\n"
                 f"Current output:\n{output}\n\n"
                 f"Feedback: {feedback}\n\n"
                 f"Produce an improved version."},
            ],
            temperature=0.3,
        )
        return resp.content

    # ──────────────────────────────────────────────────────────────
    # DREAM REALM — Context sharing
    # ──────────────────────────────────────────────────────────────

    def _gather_dream_context(self, request: str) -> str:
        """Gather relevant context from Dream Realm for a new task."""
        if not self.aperture.dream_shards:
            return ""
        # Return last N dream shards (most recent context)
        recent = self.aperture.dream_shards[-10:]
        return "\n\n".join(
            f"[{ds.source_agent}] {ds.content[:500]}" for ds in recent
        )

    # ──────────────────────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────────────────────

    def _collect_agent_names(self, km: KillerMove) -> list[str]:
        """Collect names of all agents involved."""
        names = {"fang_yuan", "star_constellation", "spectral_soul", "thieving_heaven"}
        for w in km.gu_worms:
            if w.assigned_to:
                names.add(w.assigned_to)
        return sorted(names)

    def _persist_report(self, report: ConvoyReport):
        """Save report to disk for later analysis."""
        state_dir = self.config.orchestrator.state_dir
        os.makedirs(state_dir, exist_ok=True)

        filename = f"convoy_{int(time.time())}.json"
        filepath = os.path.join(state_dir, filename)

        data = {
            "request": report.request,
            "complexity": report.complexity.value,
            "final_output": report.final_output,
            "total_llm_calls": report.total_llm_calls,
            "total_tokens": report.total_tokens,
            "total_time_seconds": report.total_time_seconds,
            "agents_used": report.agents_used,
            "iterations": report.iterations,
            "quality_scores": report.quality_scores,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("   Report saved: %s", filepath)

    def close(self):
        """Clean up resources."""
        self.llm.close()
