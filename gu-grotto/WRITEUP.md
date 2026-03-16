# Gu Refining Grotto (蛊炼洞天)

## Approximating Frontier Reasoning with Swarms of Small Models

**Version 0.5.0** — A multi-agent orchestration framework that uses coordinated swarms of small, cheap models (raptor-mini, GPT-4o-mini, Claude Haiku, Gemini Flash) to approximate the deep reasoning of frontier models (GPT-5.4-Codex, Opus 4.6, Gemini 3.1).

Named after concepts from the webnovel *Reverend Insanity* (蛊真人) by Gu Zhen Ren. Inspired by Steve Yegge's [Gas Town](https://sourcegraph.com/blog/welcome-to-gas-town) architecture for agentic orchestration.

---

## Table of Contents

1. [The Core Thesis](#the-core-thesis)
2. [Architecture Overview](#architecture-overview)
3. [Agent Roles (The Nine Venerables)](#agent-roles)
4. [Data Types (The Gu Worm Taxonomy)](#data-types)
5. [The Execution Pipeline](#the-execution-pipeline)
6. [The Myriad Self — Tournament Decomposition](#myriad-self)
7. [Quality Enforcement Chain](#quality-enforcement)
8. [Prompt Engineering for Small Models](#prompt-engineering)
9. [Design Iteration History](#iteration-history)
10. [Benchmark Suite](#benchmark-suite)
11. [Configuration & Deployment](#configuration)
12. [File Map](#file-map)
13. [Future Directions](#future-directions)

---

<a name="the-core-thesis"></a>
## 1. The Core Thesis

A single frontier model call produces a single reasoning trace. If that trace takes a wrong turn — a flawed decomposition, an incorrect assumption, a missed edge case — everything downstream inherits the error. There is no recovery.

Gu Refining Grotto inverts this: instead of one deep thinker, it deploys a *swarm* of focused small models, each handling one narrow cognitive task with structured scaffolding, reviewed by an adversarial critic, with outputs reconciled for contradictions and synthesized through iterative refinement.

The key advantages of many-small over one-big:

| Dimension | Single Frontier Model | Gu Refining Grotto |
|---|---|---|
| **Reasoning diversity** | One path | Multiple independent reasoning traces |
| **Error containment** | One error poisons everything | Failures are isolated per subtask |
| **Decomposition quality** | One plan | Tournament of competing plans (Myriad Self) |
| **Cost profile** | $0.03-0.15 per call | $0.001-0.005 per call × 15-30 calls = $0.015-0.15 |
| **Quality gates** | None (single-shot) | Per-subtask critic review + ensemble voting |
| **Parallelism** | Sequential reasoning | Parallel phase execution |

The bet: a well-orchestrated swarm of $0.001 models, with structured prompts, adversarial review, and tournament planning, can match or exceed a $0.10 model on complex tasks — at comparable cost but with better error detection and higher reliability.

---

<a name="architecture-overview"></a>
## 2. Architecture Overview

```
                    ┌──────────────────────────────┐
                    │     User Request (vague,      │
                    │     short, ambiguous)          │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │   FANG YUAN (方源)             │
                    │   Classify complexity          │
                    │   TRIVIAL → answer directly    │
                    │   SIMPLE  → single worker      │
                    │   MODERATE/COMPLEX/PROFOUND     │
                    └──────────┬───────────────────┘
                               │
               ┌───────────────▼────────────────────┐
               │  Complexity ≥ COMPLEX?              │
               │  YES → MYRIAD SELF (万我)           │
               │      3 decompositions at varying    │
               │      temperatures → probe phase 1   │
               │      → score → pick winner          │
               │  NO → STAR CONSTELLATION (星宿)     │
               │      Single decomposition           │
               └───────────────┬────────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │   SHARED ASSUMPTIONS          │
                    │   Generate architectural      │
                    │   preamble all workers share   │
                    └──────────┬───────────────────┘
                               │
        ┌──────────────────────▼──────────────────────────┐
        │               PHASE-BY-PHASE EXECUTION           │
        │                                                  │
        │  Phase 1: [Worker A] [Worker B] [Worker C]       │
        │           ↓ critic   ↓ critic   ↓ critic         │
        │           ↓ reconcile contradictions              │
        │           ↓ extractive summary for context        │
        │                                                  │
        │  Phase 2: [Worker D] [Worker E]                  │
        │           ↓ critic   ↓ critic                    │
        │           ↓ reconcile                            │
        │           ...                                    │
        └──────────────────────┬──────────────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │   THIEVING HEAVEN (偷天)       │
                    │   Hierarchical synthesis       │
                    │   Gap markers for failures     │
                    │   Iterative refinement          │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │   ENSEMBLE VOTING (3 voters)   │
                    │   Accuracy + Completeness +    │
                    │   Devil's Advocate              │
                    │   All rejection reasons used    │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │   Final Output                 │
                    └────────────────────────────────┘
```

---

<a name="agent-roles"></a>
## 3. Agent Roles (The Nine Venerables)

Each agent in the system maps to a Rank 9 Venerable from *Reverend Insanity*. The naming is deliberate — each character's *path* (cultivation specialization) maps to their orchestration role.

### Fang Yuan (方源) — The Grand Orchestrator
- **RI Role**: Protagonist. Ruthlessly efficient, uses every advantage.
- **System Role**: Receives raw user requests, classifies complexity into 5 tiers (TRIVIAL/SIMPLE/MODERATE/COMPLEX/PROFOUND), resolves ambiguity by picking the most useful interpretation.
- **Key Behavior**: Anti-trivial gate prevents short-but-complex requests from being misclassified. Requests over 10 words are promoted from TRIVIAL to SIMPLE.
- **File**: `prompts.py::FANG_YUAN_SYSTEM`

### Star Constellation (星宿) — The Wisdom Path Planner  
- **RI Role**: Rank 9 Venerable of Wisdom Path. Sees all knowledge.
- **System Role**: Decomposes tasks into self-contained subtasks with explicit dependencies and phase ordering. Performs a decomposability check — tightly-coupled tasks (LRU+TTL+thread-safe) are routed to the single-worker path.
- **Key Innovation (v3)**: Wisdom Gu — for COMPLEX/PROFOUND tasks, the decomposition call uses the frontier model, even though workers use the small model. This is the one strategic frontier-model call that has the highest leverage.
- **File**: `prompts.py::STAR_CONSTELLATION_SYSTEM`

### Spectral Soul (噬魂) — The Relentless Critic
- **RI Role**: Rank 9 Venerable of Soul Path. Devours and judges.
- **System Role**: Reviews every worker output against a structured 8-point checklist. Produces YES/NO answers; the score is computed in Python (not by the LLM). Performs root-cause analysis: blames worker, upstream context, or decomposition.
- **Key Innovation (v4)**: Backtracking — when root_cause is "upstream" or "decomposition", retries are aborted immediately. No point retrying a worker when the *task description* is wrong.
- **File**: `prompts.py::SPECTRAL_SOUL_SYSTEM`

### Thieving Heaven (偷天) — The Synthesis Master
- **RI Role**: Rank 9 Venerable of Theft Path. Steals and merges abilities.
- **System Role**: Weaves multiple worker outputs into a single coherent result. Uses hierarchical tree-reduction when outputs exceed context budget. Inserts `<!-- GAP -->` markers for failed subtasks instead of hallucinating content.
- **File**: `prompts.py::THIEVING_HEAVEN_SYSTEM`

### Paradise Earth (乐土) — The Stability Monitor
- **RI Role**: Rank 9 Venerable of Earth Path. Maintains stability.
- **System Role**: (Designed, not yet fully wired) Health monitoring — detects stuck workers, circular dependencies, and quality score trends. Recommends corrective actions: retry, reassign, split, escalate, or skip.
- **File**: `prompts.py::PARADISE_EARTH_SYSTEM`

### Giant Sun (巨阳) — The Luck Dispatcher
- **RI Role**: Rank 9 Venerable of Luck Path. Allocates fortune.
- **System Role**: (Designed, not yet fully wired) Smart task assignment — evaluates subtasks for optimal batching, priority ordering, and worker allocation.
- **File**: `prompts.py::GIANT_SUN_SYSTEM`

### Rank 5 Gu Masters — Mortal Workers
- **RI Role**: Mid-rank cultivators. Skilled but focused.
- **System Role**: Execute individual subtasks. Receive task-type-specific reasoning scaffolds (code/analysis/design/default) with structured thinking templates and confidence annotations. Each worker receives the full original request, shared assumptions, and prior phase context.
- **File**: `prompts.py::WORKER_SYSTEM`, `prompts.py::WORKER_SCAFFOLDS`

### Ensemble Voters — The Three Perspectives
- **System Role**: 3 specialized voters evaluate the final output from different angles:
  1. **Technical Accuracy** — is it factually and technically correct?
  2. **Completeness** — does it fully address the request?
  3. **Devil's Advocate** — find at least one serious problem
- **Key Fix (v4)**: All rejection reasons are concatenated and used for refinement, not just the first voter's feedback.
- **File**: `prompts.py::ENSEMBLE_VOTER_SYSTEM`

---

<a name="data-types"></a>
## 4. Data Types (The Gu Worm Taxonomy)

Every concept maps to a term from *Reverend Insanity*:

| System Concept | RI Term | Description |
|---|---|---|
| **GuWorm** (蛊虫) | Gu Worm | Atomic unit of work — a single subtask with lifecycle tracking (UNREFINED → REFINING → REFINED/REVIEWED/FAILED) |
| **KillerMove** (杀招) | Killer Move | Composite workflow — a directed graph of GuWorms with phase-ordered execution |
| **Aperture** (福地) | Blessed Land | Project workspace context — holds all Dream Shards and Dao Marks |
| **DaoMark** (道痕) | Dao Mark | Evidence of completed work — quality score, worker ID, output summary |
| **DreamShard** (梦境碎片) | Dream Shard | Shared context fragment passed between phases |
| **ConvoyReport** | Convoy | End-to-end execution report with metrics |
| **Complexity** | Rank | 5-tier classification (TRIVIAL through PROFOUND) |

### GuWorm Lifecycle

```
UNREFINED ──→ REFINING ──→ REFINED ──→ REVIEWED (critic approved)
                  │              │
                  │              └──→ REJECTED (critic rejected → retry)
                  │
                  └──→ FAILED (max attempts exhausted or root-cause non-worker)
```

---

<a name="the-execution-pipeline"></a>
## 5. The Execution Pipeline

### Phase 1: Classification (Fang Yuan)
```python
assessment = self._classify_request(request)
# Returns: complexity, summary, ambiguities, resolution, strategic_notes
```
- Temperature: 0.1 (deterministic)
- Anti-trivial gate: `if complexity == "trivial" and len(request.split()) > 10: promote to "simple"`
- TRIVIAL with `direct_answer`: return immediately (no workers)

### Phase 2: Decomposition (Star Constellation / Myriad Self)

**Standard path** (SIMPLE/MODERATE):
```python
plan = self._decompose_task(request, assessment)
```
- Decomposability check: if task is indivisible, route to `_single_worker_path()`
- Returns: KillerMove name, phases, subtasks with dependencies

**Tournament path** (COMPLEX/PROFOUND — Myriad Self):
```python
plan = self._myriad_self_decompose(request, assessment)
```
- Generates 3 decomposition plans at temperatures [0.1, 0.35, 0.6]
- Executes phase 1 of each plan as a cheap probe
- Scores probe outputs via Spectral Soul
- Selects the winning decomposition
- See [Section 6](#myriad-self) for details.

### Phase 2.5: Shared Assumptions
```python
shared_assumptions = self._generate_shared_assumptions(request, plan)
```
- Prevents architectural contradictions between parallel workers
- Bullet-point preamble: technology choices, design constraints, naming conventions

### Phase 3: Execution (Phase-by-Phase)
```python
phase_outputs = self._execute_killer_move(killer_move, plan, shared_assumptions)
```

For each phase:
1. **Parallel dispatch**: All subtasks in the phase execute simultaneously via `ThreadPoolExecutor`
2. **Worker + Critic loop** (per subtask):
   - Worker produces output using task-type scaffold
   - Spectral Soul reviews (8-point checklist)
   - If APPROVED: record Dao Mark, continue
   - If REJECTED + root_cause="worker": retry with revision feedback (up to 3 attempts)
   - If REJECTED + root_cause="upstream"/"decomposition": abort immediately (backtracking)
   - If max attempts exhausted: mark FAILED with annotation
3. **Extractive summary**: Python-based keyword extraction (no LLM call) produces context for next phase
4. **Phase reconciliation**: If multiple parallel outputs, check for contradictions

### Phase 4: Synthesis (Thieving Heaven)
```python
final_output = self._synthesize_outputs(request, plan, phase_outputs)
```

- **Standard path**: All outputs fit in context → single merge call, iteratively refined
- **Hierarchical path**: Outputs exceed context budget → tree-reduction (merge pairs → merge pairs → ... → single output)
- Failed subtasks get `<!-- GAP: ... -->` markers instead of hallucinated content

### Phase 5: Final Review (Ensemble Voting)
```python
final_output = self._final_review_loop(request, final_output)
```

- 3 specialized voters (accuracy, completeness, devil's advocate) evaluate in parallel
- Majority vote: >50% ACCEPT → ship it
- If rejected: concatenate ALL rejection reasons, refine, re-vote (up to 3 cycles)
- After max cycles: accept the best available output

---

<a name="myriad-self"></a>
## 6. The Myriad Self (万我) — Tournament Decomposition

This is the system's **game-changer** — the one capability that a single frontier model fundamentally cannot replicate.

### The Problem
In any multi-agent system, the decomposition step is the highest-leverage decision. A bad decomposition produces bad subtasks, which produce bad outputs, which no amount of critic review can fix. Yet every existing framework (AutoGen, CrewAI, LangGraph) uses a *single* decomposition pass.

### The Insight
Multiple small models, called with different temperatures, produce *genuinely different* decomposition strategies. We can cheaply probe each strategy by executing only its first phase, then commit to the winner.

This is **beam search at the planning level** — something CoT and repeated sampling within a single model cannot achieve, because the model's internal state is opaque and cannot be diverged/merged.

### The Mechanism

```
Request → STAR CONSTELLATION called 3× at temps [0.1, 0.35, 0.6]
  │
  ├── Plan A: "Infrastructure-First" (DB schema → API → Frontend → Tests)
  ├── Plan B: "Feature-Slice" (Auth slice → Profile slice → Admin slice)
  └── Plan C: "Risk-First" (Hardest problem → Integration → Polish)
  │
  │ Execute PHASE 1 ONLY of each plan (cheap: 1-3 workers per plan)
  │
  ├── Plan A Phase 1 outputs → Spectral Soul scores → 0.72
  ├── Plan B Phase 1 outputs → Spectral Soul scores → 0.85
  └── Plan C Phase 1 outputs → Spectral Soul scores → 0.61
  │
  └── 🏆 Winner: Plan B (score 0.85)
      → Execute remaining phases of Plan B
```

### Cost Analysis
For a COMPLEX task that would normally use ~20 small-model calls:
- **Extra calls for tournament**: 3 decompositions + 3-9 phase-1 workers + 3-9 critic reviews = ~9-21 extra calls
- **Percentage overhead**: ~45-100%
- **When triggered**: Only for COMPLEX/PROFOUND tasks where the decomposition has high leverage
- **SIMPLE/MODERATE tasks**: Skip directly to single decomposition (no overhead)

The 45-100% overhead buys qualitatively different planning diversity — cheap insurance against the most common failure mode of multi-agent systems.

---

<a name="quality-enforcement"></a>
## 7. Quality Enforcement Chain

The system has **5 independent quality enforcement mechanisms**, ensuring failures are caught at multiple levels:

### 1. Checklist-Based Critic (Per Subtask)
```json
{
  "addresses_task": true,       // Does it answer what was asked?
  "factual_errors": false,      // Any incorrect claims?
  "format_correct": true,       // Right output format?
  "all_requirements": true,     // All requirements covered?
  "specific_enough": true,      // Concrete, not vague?
  "would_embarrass": false,     // Quality bar?
  "concurrent_safety": true,    // Thread-safe? (code only)
  "edge_cases": true            // Edge cases handled? (code only)
}
```
Score is computed in Python: `score = correct_answers / total_checks`. The LLM answers YES/NO; arithmetic is never delegated to the model.

### 2. Root-Cause Analysis (Per Subtask)
When a critic rejects output, it determines *why*:
- **worker**: The worker just did a bad job → retry
- **upstream**: A previous phase gave bad context → abort retries (backtracking)
- **decomposition**: The subtask itself is mis-specified → abort retries

This prevents the most expensive failure mode: repeatedly retrying a worker when the *task description* is the problem.

### 3. Phase Reconciliation (Between Parallel Subtasks)
After each phase, parallel outputs are checked for contradictions. If Worker A chose PostgreSQL and Worker B chose MongoDB, the reconciliation warning is injected into subsequent context.

### 4. Gap Markers (In Synthesis)
Failed subtasks are explicitly annotated in the final output:
```html
<!-- GAP: [subtask name] could not be completed. -->
```
The synthesizer is instructed to *never* hallucinate content to fill gaps. Honesty over completeness.

### 5. Ensemble Voting (Final Output)
Three specialized voters evaluate from different angles. The **Devil's Advocate** voter is required to find at least one serious problem — if it can't, it must explain why in detail. This asymmetry ensures the voting is not a rubber stamp.

### Fail-Closed Philosophy
Every quality gate defaults to rejection on parse failure:
- JSON parse failure → `REJECTED`
- Missing `verdict` field → `REJECTED`
- Missing `vote` field → `REJECT`
- Missing `overall_score` → `0.0`

The system never silently accepts garbage.

---

<a name="prompt-engineering"></a>
## 8. Prompt Engineering for Small Models

Small models require fundamentally different prompting than frontier models. Here are the principles refined through 4 design iterations:

### Principle 1: One Cognitive Task Per Agent
Each prompt constrains the agent to exactly ONE thinking operation:
- Fang Yuan: classify, don't solve
- Star Constellation: decompose, don't execute
- Spectral Soul: evaluate, don't fix
- Workers: execute, don't plan

This prevents the common failure mode where small models attempt to do everything at once and do everything poorly.

### Principle 2: Structured Output Schemas
Every agent prompt includes a JSON schema for its response. This is not optional guidance — it's the entire contract. Parse failures are treated as rejections.

### Principle 3: Reasoning Scaffolds
Workers receive task-type-specific thinking templates:

**Code scaffold**:
```
1. RESTATE the exact coding task
2. CONSTRAINTS I must satisfy
3. PRECONDITIONS (inputs with types)
4. POSTCONDITIONS (outputs with types)
5. INVARIANTS that must always hold
6. ALGORITHM approach
7. EDGE CASES to handle
8. IMPLEMENTATION
```

**Analysis scaffold**:
```
1. RESTATE the analysis question
2. THESIS (main position)
3. EVIDENCE FOR
4. EVIDENCE AGAINST
5. TRADEOFFS
6. SYNTHESIS
7. RECOMMENDATION
```

Each claim is annotated with confidence: `[CERTAIN]`, `[LIKELY]`, or `[SPECULATIVE]`.

### Principle 4: Binary Gates
Complex judgments are reduced to YES/NO checklist items. The human tendency of LLMs to "hedge" and produce ambiguous evaluations is eliminated. The model answers binary questions; the system computes the score.

### Principle 5: Scaffold Selection
The system detects task type from subtask description keywords:
- `"implement"`, `"function"`, `"class"`, `"algorithm"` → code scaffold
- `"analyze"`, `"evaluate"`, `"compare"`, `"review"` → analysis scaffold
- `"design"`, `"architect"`, `"plan"`, `"propose"` → design scaffold
- Everything else → default scaffold

---

<a name="iteration-history"></a>
## 9. Design Iteration History

The system went through **4 full design iterations**, each analyzed by a dedicated research subagent that produced 300-700 line analyses of weaknesses. This section documents what was broken and what fixed it.

### V1 → V2: The Wiring Fixes

**V1 weaknesses identified** (8 categories, 23 specific issues):

| Category | Issue | Impact |
|---|---|---|
| Classification bottleneck | Single LLM classifies complexity; if wrong, irrecoverable | Entire convoy fails |
| JSON hallucination | LLM-reported scores unreliable (arithmetic errors, anchoring) | Silent quality degradation |
| Context window saturation | Raw outputs concatenated into downstream prompts | LLM confusion, dropped content |
| Assumption divergence | Parallel workers make contradictory architectural choices | Incoherent synthesis |
| Auto-approval cascade | No fail-closed defaults on parse failure | Garbage passes quality gates |
| Lossy compression | Full outputs truncated by character count | Important details lost |
| Ensemble correlation | All voters see same prompt, produce correlated votes | Fake consensus |
| Decomposition paradox | Tightly-coupled tasks decomposed into incoherent subtasks | Impossible subtasks |

**V2 fixes applied**:
1. Python-computed scores (no LLM arithmetic)
2. Fail-closed defaults (REJECTED on parse failure)
3. Full original request propagated to all workers
4. Decomposability check + single-worker path
5. Hierarchical tree-reduction synthesis
6. Specialized ensemble voters (different evaluation axes)
7. Shared assumptions preamble for parallel workers
8. FAILED marking on exhausted retries (not silently accepted)
9. Extractive summaries between phases (reduced context bloat)
10. Phase reconciliation for contradiction detection

### V2 → V3: The Reasoning Scaffolds

**V2 weaknesses identified** (6 categories):

| Category | Issue | Fix |
|---|---|---|
| Unstructured worker reasoning | Workers free-associate, miss edge cases | Task-type-specific scaffolds with confidence annotations |
| Opaque critic scoring | Critic produces gestures, not checkable claims | 8-point YES/NO checklist |
| Blind retry loop | Worker retried even when task description is the problem | Root-cause analysis: worker/upstream/decomposition |
| LLM-heavy context handoff | Summarization used an LLM call per phase | Python extractive summary (keyword-based) |
| Decomposition quality | Small model makes poor decomposition decisions on complex tasks | Wisdom Gu: frontier model for decomposition of complex/profound tasks |
| Redundant ensemble | 3 voters but only marginal benefit over 2 | Reduced to 2 voters (restored to 3 in v4) |

### V3 → V4: The Dead Code Wiring + Game-Changer

**V3 weaknesses identified** (critical: dead code that was designed but never connected):

| Issue | Fix |
|---|---|
| `WORKER_SCAFFOLDS` defined but never injected into `_run_worker` | Wired: keyword detection selects scaffold and appends to system prompt |
| Backtracking designed but critic `root_cause`/`backtrack_to` never read | Wired: non-worker root causes abort retries immediately |
| Ensemble uses only first voter's feedback for refinement | Fixed: all rejection reasons concatenated |
| `overall_score` can be undefined on empty checklist | Fixed: `result.setdefault("overall_score", 0.0)` |
| Single decomposition path — no plan diversity | **Myriad Self**: tournament decomposition with 3 candidates probed at phase 1 |
| Ensemble reduced to 2 voters (lost devil's advocate) | Restored to 3 voters |

### Iteration Methodology

Each iteration followed the same protocol:
1. **Research subagent** (dual-agent adversarial analysis) produces a 300-700 line report of weaknesses
2. **Prioritization** into P0 (must-fix), P1 (should-fix), P2 (nice-to-have)
3. **Implementation** of all P0 and selected P1 items
4. **Syntax validation** via `ast.parse()` on all files
5. **Architectural diff** against the original design

---

<a name="benchmark-suite"></a>
## 10. Benchmark Suite

9 synthetic tasks across 4 tiers, designed to stress different orchestration capabilities:

### Tier 1: Decomposition & Parallel Execution
| Task | Request | Tests |
|---|---|---|
| `arch-001` | "design a distributed caching layer" | Systems decomposition (eviction, consistency, partitioning, API) |
| `code-001` | "implement rate limiting" | Vague request + code generation (must infer language, algorithm) |
| `analysis-001` | "when should I use microservices" | Balanced analysis (must present both sides) |

### Tier 2: Synthesis & Quality Gates
| Task | Request | Tests |
|---|---|---|
| `synth-001` | "write an architecture doc for a real-time collaborative editor" | Comprehensive synthesis (CRDT/OT, WebSockets, data model) |
| `debug-001` | "Our API has intermittent 500 errors..." | Systematic debugging methodology |

### Tier 3: Adversarial Review & Refinement
| Task | Request | Tests |
|---|---|---|
| `algo-001` | "LRU cache with TTL, thread-safe, Python" | Multi-constraint code (LRU + TTL + concurrency) |
| `multi-001` | "We're migrating from Django monolith to microservices" | Multi-domain planning (infra, code, data, process) |

### Tier 4: Extreme Ambiguity
| Task | Request | Tests |
|---|---|---|
| `ambig-001` | "it's slow" | Maximal ambiguity — 2 words, must infer everything |
| `ambig-002` | "improve the auth" | Moderate ambiguity — must infer system and "improve" meaning |

### Benchmark Runner
```bash
# Run full benchmark suite
python run.py --benchmark

# Compare against any frontier model
GROTTO_FRONTIER_MODEL=gpt-4o GROTTO_FRONTIER_API_KEY=sk-... python run.py --benchmark
```

The runner executes each task through both the orchestrated pipeline and a direct frontier model call, then uses a meta-evaluator to compare outputs on 5 axes (correctness, completeness, depth, clarity, actionability), each scored 1-10.

---

<a name="configuration"></a>
## 11. Configuration & Deployment

### Quick Start

```bash
cd gu-grotto
pip install -r requirements.txt   # just httpx>=0.27.0

# Set your API key (any OpenAI-compatible endpoint)
export GROTTO_API_KEY="sk-..."
export GROTTO_API_BASE="https://api.openai.com/v1"  # or any compatible endpoint
export GROTTO_MODEL="gpt-4o-mini"                    # or raptor-mini, claude-haiku, etc.

# Run a request
python run.py "design a rate limiter for a multi-tenant SaaS API"

# Run benchmarks
python run.py --benchmark
```

### Configuration File

```json
{
  "model": {
    "model": "raptor-mini",
    "api_base": "https://api.openai.com/v1",
    "temperature": 0.3,
    "max_tokens": 4096,
    "frontier_model": "gpt-4o",
    "frontier_api_base": "https://api.openai.com/v1"
  },
  "orchestrator": {
    "max_workers": 12,
    "max_review_cycles": 3,
    "quality_threshold": 0.7,
    "ensemble_voting": true,
    "ensemble_size": 3,
    "adversarial_review": true,
    "synthesis_iterations": 2,
    "wisdom_gu_enabled": true,
    "backtracking_enabled": true,
    "max_backtrack_depth": 2,
    "myriad_self_enabled": true,
    "myriad_self_candidates": 3
  }
}
```

```bash
python run.py --config config.json "your request"
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROTTO_API_KEY` | (none) | API key for the small model |
| `GROTTO_API_BASE` | `https://api.openai.com/v1` | API endpoint |
| `GROTTO_MODEL` | `raptor-mini` | Small model name |
| `GROTTO_MAX_WORKERS` | `12` | Maximum parallel workers |
| `GROTTO_VERBOSE` | `1` | Enable verbose logging |
| `GROTTO_FRONTIER_MODEL` | `gpt-4o` | Frontier model for Wisdom Gu + benchmarks |
| `GROTTO_FRONTIER_API_KEY` | (same as `GROTTO_API_KEY`) | Frontier model API key |

### Key Tuning Knobs

| Parameter | Effect of Increasing | Cost Impact |
|---|---|---|
| `max_workers` | More parallelism, faster phases | No extra LLM calls |
| `max_review_cycles` | More critic iterations per subtask | +1-2 calls per subtask per cycle |
| `quality_threshold` | Higher bar for approval (0.0-1.0) | More retries |
| `ensemble_size` | More diverse final review | +N calls per review cycle |
| `synthesis_iterations` | More polished synthesis | +1 call per iteration |
| `myriad_self_candidates` | More decomposition diversity | +N decompositions + N×phase1 probes |

### Using with Different LLM Providers

The system works with any OpenAI-compatible API:

```bash
# Ollama (local)
export GROTTO_API_BASE="http://localhost:11434/v1"
export GROTTO_MODEL="llama3.2:3b"
export GROTTO_API_KEY="ollama"

# Together AI
export GROTTO_API_BASE="https://api.together.xyz/v1"
export GROTTO_MODEL="meta-llama/Llama-3-8b-chat-hf"

# Groq (fast inference)
export GROTTO_API_BASE="https://api.groq.com/openai/v1"
export GROTTO_MODEL="llama-3.1-8b-instant"
```

---

<a name="file-map"></a>
## 12. File Map

```
gu-grotto/
├── run.py                      # CLI entry point (argparse)
├── config.example.json         # Example configuration file
├── requirements.txt            # Dependencies (httpx>=0.27.0)
│
├── grotto/
│   ├── __init__.py             # Package init, version (0.5.0)
│   ├── types.py                # Core data types (GuWorm, KillerMove, Aperture, etc.)
│   ├── config.py               # Configuration management (env vars + JSON file)
│   ├── llm.py                  # Model-agnostic LLM client (httpx, retry, JSON parsing)
│   ├── prompts.py              # All agent prompts + reasoning scaffolds
│   └── orchestrator.py         # Main orchestration engine (Fang Yuan)
│
└── benchmarks/
    ├── __init__.py
    ├── tasks.py                # 9 synthetic benchmark task definitions
    └── runner.py               # Benchmark execution + comparison framework
```

### Module Dependency Graph

```
run.py
  └── grotto/
      ├── orchestrator.py
      │   ├── config.py → ModelConfig, OrchestratorConfig, GrottoConfig
      │   ├── llm.py → LLMClient, LLMResponse
      │   ├── types.py → GuWorm, KillerMove, Aperture, DaoMark, DreamShard, ConvoyReport
      │   └── prompts.py → all prompt constants
      │
      └── benchmarks/runner.py
          ├── orchestrator.py → GrottoOrchestrator
          ├── prompts.py → META_EVALUATOR_SYSTEM
          └── benchmarks/tasks.py → BENCHMARK_TASKS
```

---

<a name="future-directions"></a>
## 13. Future Directions

### P0: Wire Remaining Agents
- **Paradise Earth**: Health monitoring with stuck-worker detection and corrective recommendations
- **Giant Sun**: Smart dispatch with priority ordering and batch optimization

### P1: Calamity and Tribulation (劫难) — Adversarial Stress Testing  
After synthesis, a dedicated "Calamity" agent generates the hardest possible counter-argument, test case, or failure scenario for the output. Unlike the critic (which evaluates), this agent actively *attacks*. If the output survives the Calamity, it's genuinely robust.

### P1: Fixed Immortal Travel (定仙游) — Deterministic Replay
Full trace serialization: every LLM call, its input/output, timing, and scoring → a DAG that can be replayed, inspected, and diffed. Essential for debugging why a run produced poor output.

### P2: Async Execution
Replace `ThreadPoolExecutor` + `httpx.Client` with `asyncio` + `httpx.AsyncClient`. Add per-worm timeouts. Fix the `_original_request` thread-safety issue (currently stored as instance attribute).

### P2: Mock LLM for CI  
Deterministic mock mode where `LLMClient` returns canned responses for specific prompt hashes. Enables testing orchestration logic without API costs, plus CI/CD integration.

### P3: Cost Tracking per Phase
Tag every LLM call with `(phase, agent, worm_id)`. Emit structured JSON logs. Identify which phase is the cost bottleneck.

### P3: Learning from Critic Patterns
If the same failure type recurs (e.g., "missing edge cases" on every code task), adapt worker prompts or scaffolds dynamically.

---

## Appendix: The Gas Town Lineage

This system owes its architectural DNA to Steve Yegge's [Gas Town](https://sourcegraph.com/blog/welcome-to-gas-town) concept. Key mappings:

| Gas Town Concept | Gu Refining Grotto | 
|---|---|
| Mayor | Fang Yuan (orchestrator) |
| Polecats | Rank 5 Gu Masters (workers) |
| Refinery | Thieving Heaven (synthesis) |
| Witness | Paradise Earth (monitoring) |
| Deacon | Giant Sun (dispatch) |
| Dogs (adversarial) | Spectral Soul (critic) |
| Convoy | ConvoyReport (end-to-end execution) |
| Beads | GuWorms (atomic work units) |
| Molecules | KillerMoves (composite workflows) |
| MEOW stack | Gu Worm taxonomy |
| NDI (nondeterministic idempotence) | Myriad Self (tournament exploration) |

The primary innovation over Gas Town is the **Myriad Self tournament decomposition**, which exploits the specific advantage of many-small-model orchestration: cheap exploration of parallel planning hypotheses followed by empirical selection. This is beam search at the planning level — enabled by the economic structure of cheap model inference, which makes probing multiple plans viable.

---

*"This sovereign is willing to refine any Gu, endure any tribulation, as long as it leads to eternal life."*
— Fang Yuan, Reverend Insanity

---

**License**: MIT  
**Version**: 0.5.0  
**Design iterations**: 4 (v1 → v2 → v3 → v4)  
**Total research agent analyses**: 4 (1,800+ lines of structured criticism)  
**Agents in the system**: 7 defined (5 fully wired, 2 designed)  
**Quality gates**: 5 independent enforcement mechanisms  
**Benchmark tasks**: 9 across 4 difficulty tiers
