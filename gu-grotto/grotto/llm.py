"""
LLM client — model-agnostic adapter for any OpenAI-compatible API.

Supports raptor-mini, GPT-4o-mini, Claude Haiku, Gemini Flash, or any
model served via an OpenAI-compatible endpoint (vLLM, Ollama, LiteLLM, etc.)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import ModelConfig

logger = logging.getLogger("grotto.llm")


@dataclass
class LLMResponse:
    """Wrapper for LLM response with metadata."""
    content: str = ""
    model: str = ""
    tokens_prompt: int = 0
    tokens_completion: int = 0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def tokens_total(self) -> int:
        return self.tokens_prompt + self.tokens_completion


class LLMClient:
    """
    Synchronous LLM client using httpx.
    Handles retries, structured output parsing, and token tracking.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.total_calls = 0
        self.total_tokens = 0
        self._client = httpx.Client(timeout=config.timeout)

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request."""
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        # Determine which API base/key to use
        if model == self.config.frontier_model:
            api_base = self.config.frontier_api_base
            api_key = self.config.frontier_api_key
        else:
            api_base = self.config.api_base
            api_key = self.config.api_key

        url = f"{api_base.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        start = time.monotonic()
        resp = self._make_request(url, headers, payload)
        latency = (time.monotonic() - start) * 1000

        content = ""
        tokens_prompt = 0
        tokens_completion = 0

        if "choices" in resp and resp["choices"]:
            content = resp["choices"][0].get("message", {}).get("content", "")
        if "usage" in resp:
            tokens_prompt = resp["usage"].get("prompt_tokens", 0)
            tokens_completion = resp["usage"].get("completion_tokens", 0)

        self.total_calls += 1
        self.total_tokens += tokens_prompt + tokens_completion

        result = LLMResponse(
            content=content,
            model=model,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            latency_ms=latency,
            raw=resp,
        )

        logger.debug(
            "LLM call: model=%s tokens=%d latency=%.0fms",
            model, result.tokens_total, latency,
        )
        return result

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict:
        """Chat and parse response as JSON. Falls back to extraction on parse failure."""
        resp = self.chat(messages, model=model, temperature=temperature, json_mode=True)
        try:
            return json.loads(resp.content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            content = resp.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            try:
                return json.loads(content.strip())
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from LLM response, returning raw")
                return {"raw": resp.content}

    def _make_request(self, url: str, headers: dict, payload: dict, retries: int = 3) -> dict:
        """Make HTTP request with retry logic."""
        last_error = None
        for attempt in range(retries):
            try:
                response = self._client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    wait = min(2 ** attempt * 2, 30)
                    logger.warning("Rate limited, waiting %ds (attempt %d/%d)", wait, attempt + 1, retries)
                    time.sleep(wait)
                elif e.response.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning("Server error %d, retrying in %ds", e.response.status_code, wait)
                    time.sleep(wait)
                else:
                    raise
            except httpx.RequestError as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning("Request error: %s, retrying in %ds", e, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM request failed after {retries} retries: {last_error}")

    def close(self):
        self._client.close()
