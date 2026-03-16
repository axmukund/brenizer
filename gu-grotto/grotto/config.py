"""
Configuration for the Gu Refining Grotto.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the LLM backend."""
    model: str = "raptor-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 120

    # Frontier model for benchmark comparison
    frontier_model: str = "gpt-4o"
    frontier_api_base: str = "https://api.openai.com/v1"
    frontier_api_key: str = ""


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestration engine."""
    # Maximum parallel workers
    max_workers: int = 12
    # Maximum critic review iterations per worm
    max_review_cycles: int = 3
    # Quality threshold (0-1) for critic approval
    quality_threshold: float = 0.7
    # Whether to use ensemble voting for ambiguous tasks
    ensemble_voting: bool = True
    # Number of ensemble voters
    ensemble_size: int = 3  # v4: restored to 3 — accuracy + completeness + devil's advocate
    # Whether to use adversarial review
    adversarial_review: bool = True
    # Progressive refinement iterations for synthesis
    synthesis_iterations: int = 2
    # Enable verbose logging
    verbose: bool = True
    # Persist state to JSON
    persist_state: bool = True
    # State directory
    state_dir: str = "./grotto_state"
    # v3: Wisdom Gu — use frontier model for decomposition on complex tasks
    wisdom_gu_enabled: bool = True
    # v3: Enable backtracking on critic root-cause=upstream/decomposition
    backtracking_enabled: bool = True
    # v3: Maximum backtrack depth (prevent infinite loops)
    max_backtrack_depth: int = 2
    # v4: Myriad Self — tournament decomposition for complex/profound tasks
    myriad_self_enabled: bool = True
    # v4: Number of candidate decompositions to generate
    myriad_self_candidates: int = 3


@dataclass
class GrottoConfig:
    """Top-level configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    @classmethod
    def from_env(cls) -> GrottoConfig:
        """Load configuration from environment variables."""
        cfg = cls()
        cfg.model.api_key = os.environ.get("GROTTO_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        cfg.model.api_base = os.environ.get("GROTTO_API_BASE", cfg.model.api_base)
        cfg.model.model = os.environ.get("GROTTO_MODEL", cfg.model.model)
        cfg.model.frontier_api_key = os.environ.get("GROTTO_FRONTIER_API_KEY", cfg.model.api_key)
        cfg.model.frontier_model = os.environ.get("GROTTO_FRONTIER_MODEL", cfg.model.frontier_model)
        cfg.orchestrator.max_workers = int(os.environ.get("GROTTO_MAX_WORKERS", cfg.orchestrator.max_workers))
        cfg.orchestrator.verbose = os.environ.get("GROTTO_VERBOSE", "1") == "1"
        return cfg

    @classmethod
    def from_file(cls, path: str) -> GrottoConfig:
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        cfg = cls()
        if "model" in data:
            for k, v in data["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        if "orchestrator" in data:
            for k, v in data["orchestrator"].items():
                if hasattr(cfg.orchestrator, k):
                    setattr(cfg.orchestrator, k, v)
        # Always overlay env vars for secrets
        if not cfg.model.api_key:
            cfg.model.api_key = os.environ.get("GROTTO_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        if not cfg.model.frontier_api_key:
            cfg.model.frontier_api_key = os.environ.get("GROTTO_FRONTIER_API_KEY", cfg.model.api_key)
        return cfg
