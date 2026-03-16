"""
Synthetic benchmark tasks for evaluating the Gu Refining Grotto.

Each task is designed to test a different aspect of the orchestration:
1. Deep reasoning / multi-step logic
2. Broad knowledge synthesis
3. Code generation with edge cases
4. Ambiguous request interpretation
5. Creative / open-ended tasks

Tasks are categorized by expected complexity tier.
"""

BENCHMARK_TASKS = [
    # ──────────────────────────────────────────────────────────────
    # TIER 1: Tests decomposition and parallel execution
    # ──────────────────────────────────────────────────────────────
    {
        "id": "arch-001",
        "name": "Distributed Cache Design",
        "request": "design a distributed caching layer",
        "description": "Tests: decomposition of a systems design problem into "
                       "sub-components (eviction, consistency, partitioning, API)",
        "expected_complexity": "complex",
        "evaluation_criteria": [
            "Covers eviction policies",
            "Addresses consistency models",
            "Discusses partitioning/sharding",
            "Defines clear API",
            "Considers failure modes",
            "Mentions monitoring/observability",
        ],
    },
    {
        "id": "code-001",
        "name": "Rate Limiter Implementation",
        "request": "implement rate limiting",
        "description": "Tests: vague request interpretation + code generation. "
                       "Must infer language, choose algorithm, handle edge cases.",
        "expected_complexity": "moderate",
        "evaluation_criteria": [
            "Chooses a reasonable algorithm (token bucket, sliding window, etc.)",
            "Provides working code",
            "Handles concurrent access",
            "Includes configuration options",
            "Has clear API",
        ],
    },
    {
        "id": "analysis-001",
        "name": "Microservices vs Monolith Tradeoff Analysis",
        "request": "when should I use microservices",
        "description": "Tests: nuanced analysis requiring balanced perspective. "
                       "Should not be one-sided.",
        "expected_complexity": "moderate",
        "evaluation_criteria": [
            "Presents both sides fairly",
            "Identifies specific decision criteria",
            "Gives concrete examples",
            "Discusses organizational factors",
            "Mentions migration strategies",
        ],
    },
    # ──────────────────────────────────────────────────────────────
    # TIER 2: Tests synthesis and quality gates
    # ──────────────────────────────────────────────────────────────
    {
        "id": "synth-001",
        "name": "Full-Stack Architecture Document",
        "request": "write an architecture doc for a real-time collaborative editor",
        "description": "Tests: comprehensive document generation requiring "
                       "knowledge of CRDT/OT, WebSockets, databases, deployment.",
        "expected_complexity": "complex",
        "evaluation_criteria": [
            "Covers real-time sync mechanism (CRDT or OT)",
            "WebSocket or similar transport",
            "Data model design",
            "Conflict resolution strategy",
            "Scalability considerations",
            "Security model",
            "Deployment architecture",
            "Client-side architecture",
        ],
    },
    {
        "id": "debug-001",
        "name": "Root Cause Analysis",
        "request": (
            "Our API has intermittent 500 errors, roughly 0.1% of requests, "
            "happening more during peak hours. The service runs on Kubernetes "
            "with 3 replicas behind an nginx ingress. Database is PostgreSQL. "
            "What should we investigate?"
        ),
        "description": "Tests: systematic debugging methodology. "
                       "Must be structured and actionable.",
        "expected_complexity": "moderate",
        "evaluation_criteria": [
            "Systematic investigation approach",
            "Considers connection pool exhaustion",
            "Considers resource limits (CPU/memory)",
            "Suggests specific diagnostic commands/queries",
            "Considers database-level issues",
            "Mentions load-dependent failure modes",
        ],
    },
    # ──────────────────────────────────────────────────────────────
    # TIER 3: Tests adversarial review and iterative refinement
    # ──────────────────────────────────────────────────────────────
    {
        "id": "algo-001",
        "name": "LRU Cache with TTL",
        "request": "LRU cache that also supports TTL per entry, thread-safe, in Python",
        "description": "Tests: precise algorithmic implementation with multiple "
                       "constraints that must all be satisfied simultaneously.",
        "expected_complexity": "moderate",
        "evaluation_criteria": [
            "Correct LRU eviction behavior",
            "TTL expiration works correctly",
            "Thread-safety with proper locking",
            "O(1) get and put operations",
            "Clean API (get, put, delete)",
            "Handles edge cases (expired entries, capacity 0)",
        ],
    },
    {
        "id": "multi-001",
        "name": "Multi-Domain Technical Plan",
        "request": (
            "We're migrating from a monolithic Django app to microservices. "
            "Give us a migration plan."
        ),
        "description": "Tests: broad, multi-domain planning requiring coordination "
                       "across infrastructure, code, data, and process changes.",
        "expected_complexity": "profound",
        "evaluation_criteria": [
            "Phased migration approach (strangler fig or similar)",
            "Service boundary identification methodology",
            "Data migration strategy",
            "API gateway / service mesh",
            "CI/CD pipeline changes",
            "Monitoring and observability",
            "Team organization changes",
            "Rollback strategy",
            "Timeline estimation methodology",
        ],
    },
    # ──────────────────────────────────────────────────────────────
    # TIER 4: Tests the full pipeline under extreme ambiguity
    # ──────────────────────────────────────────────────────────────
    {
        "id": "ambig-001",
        "name": "Fix the Performance",
        "request": "it's slow",
        "description": "Tests: maximal ambiguity resolution. The orchestrator must "
                       "infer a useful interpretation from two words.",
        "expected_complexity": "moderate",
        "evaluation_criteria": [
            "Interprets 'it' reasonably (web app, API, database, etc.)",
            "Provides a systematic performance investigation approach",
            "Covers multiple performance domains",
            "Gives actionable diagnostic steps",
        ],
    },
    {
        "id": "ambig-002",
        "name": "Make It Better",
        "request": "improve the auth",
        "description": "Tests: moderate ambiguity. Must infer what 'auth' system "
                       "and what 'improve' means.",
        "expected_complexity": "moderate",
        "evaluation_criteria": [
            "Identifies multiple auth improvement areas",
            "Covers security hardening",
            "Considers UX improvements",
            "Provides specific, actionable recommendations",
        ],
    },
]


def get_task(task_id: str) -> dict | None:
    """Get a benchmark task by ID."""
    return next((t for t in BENCHMARK_TASKS if t["id"] == task_id), None)


def get_tasks_by_complexity(complexity: str) -> list[dict]:
    """Get benchmark tasks filtered by expected complexity."""
    return [t for t in BENCHMARK_TASKS if t["expected_complexity"] == complexity]
