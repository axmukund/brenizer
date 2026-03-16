"""
Core types for the Gu Refining Grotto.

Naming convention from Reverend Insanity:
- GuWorm (蛊虫): Atomic unit of work — a single task
- KillerMove (杀招): Composite workflow — a sequence/graph of GuWorms
- Aperture (福地): Project workspace context
- DaoMark (道痕): Evidence/trace of completed work
- DreamShard (梦境碎片): Shared context fragment
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Rank(Enum):
    """Agent capability tier. Higher rank = more authority."""
    MORTAL_ONE = 1    # Disposable micro-worker (single focused task)
    MORTAL_THREE = 3  # Standard worker (moderate complexity)
    MORTAL_FIVE = 5   # Skilled worker (can plan within scope)
    IMMORTAL_SIX = 6  # Persistent specialist (cross-task context)
    VENERABLE_NINE = 9  # System-level orchestrator


class GuWormStatus(Enum):
    """Lifecycle of an atomic work unit."""
    UNREFINED = "unrefined"        # Created but not started
    REFINING = "refining"          # In progress
    REFINED = "refined"            # Completed successfully
    FAILED = "failed"              # Failed, needs retry or escalation
    REVIEWED = "reviewed"          # Passed critic review
    REJECTED = "rejected"          # Failed critic review, needs rework


class Complexity(Enum):
    """Task complexity classification."""
    TRIVIAL = "trivial"      # Single worker, no decomposition needed
    SIMPLE = "simple"        # 1-2 workers, minimal decomposition
    MODERATE = "moderate"    # 3-5 workers, clear decomposition
    COMPLEX = "complex"      # 6-12 workers, multi-phase decomposition
    PROFOUND = "profound"    # 13+ workers, hierarchical decomposition


@dataclass
class GuWorm:
    """
    Atomic unit of work. Like a Bead in Gas Town.
    Named after the magical parasites in Reverend Insanity.
    """
    id: str = field(default_factory=lambda: f"gu-{uuid.uuid4().hex[:8]}")
    title: str = ""
    description: str = ""
    status: GuWormStatus = GuWormStatus.UNREFINED
    assigned_to: str | None = None
    parent_killer_move: str | None = None
    dependencies: list[str] = field(default_factory=list)
    output: str = ""
    critique: str = ""
    attempt: int = 0
    max_attempts: int = 3
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    context_fragments: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "parent_killer_move": self.parent_killer_move,
            "dependencies": self.dependencies,
            "output": self.output,
            "critique": self.critique,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "context_fragments": self.context_fragments,
            "metadata": self.metadata,
        }


@dataclass
class KillerMove:
    """
    Composite workflow — a directed graph of GuWorms.
    Like a Molecule in Gas Town. Named after the combined Gu techniques.
    """
    id: str = field(default_factory=lambda: f"km-{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    gu_worms: list[GuWorm] = field(default_factory=list)
    execution_order: list[list[str]] = field(default_factory=list)  # Phases: [[parallel ids], [parallel ids], ...]
    status: str = "pending"
    final_output: str = ""
    created_at: float = field(default_factory=time.time)

    def get_ready_worms(self) -> list[GuWorm]:
        """Return worms whose dependencies are all satisfied."""
        completed_ids = {
            w.id for w in self.gu_worms
            if w.status in (GuWormStatus.REFINED, GuWormStatus.REVIEWED)
        }
        return [
            w for w in self.gu_worms
            if w.status == GuWormStatus.UNREFINED
            and all(dep in completed_ids for dep in w.dependencies)
        ]


@dataclass
class Aperture:
    """
    Project workspace context. Like a Rig in Gas Town.
    Named after the blessed lands where Gu Immortals cultivate.
    """
    id: str = field(default_factory=lambda: f"ap-{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    active_killer_moves: list[KillerMove] = field(default_factory=list)
    dream_shards: list[DreamShard] = field(default_factory=list)
    dao_marks: list[DaoMark] = field(default_factory=list)


@dataclass
class DaoMark:
    """
    Evidence of completed work — an immutable record.
    Named after the cultivation traces left by path masters.
    """
    id: str = field(default_factory=lambda: f"dm-{uuid.uuid4().hex[:8]}")
    gu_worm_id: str = ""
    worker_id: str = ""
    output_summary: str = ""
    quality_score: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class DreamShard:
    """
    Shared context fragment accessible to all agents in an Aperture.
    Named after the Dream Realm where cultivators share experiences.
    """
    id: str = field(default_factory=lambda: f"ds-{uuid.uuid4().hex[:8]}")
    content: str = ""
    source_agent: str = ""
    relevance_tags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentIdentity:
    """Persistent agent identity — survives across sessions."""
    id: str = ""
    name: str = ""
    role: str = ""
    rank: Rank = Rank.MORTAL_THREE
    system_prompt: str = ""
    active: bool = True
    tasks_completed: int = 0
    tasks_failed: int = 0


@dataclass
class ConvoyReport:
    """
    Final report from a completed convoy (end-to-end task execution).
    Like a Convoy in Gas Town.
    """
    request: str = ""
    complexity: Complexity = Complexity.SIMPLE
    killer_move: KillerMove | None = None
    final_output: str = ""
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    agents_used: list[str] = field(default_factory=list)
    iterations: int = 0
    quality_scores: list[float] = field(default_factory=list)
    dao_marks: list[DaoMark] = field(default_factory=list)
