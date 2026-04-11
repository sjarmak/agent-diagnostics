"""Deterministic fake LLM backend for testing the annotation pipeline.

Provides ``FakeLLMBackend`` — a drop-in replacement for real LLM calls
that returns pre-defined, deterministic responses keyed by category names
found in the input prompt.  Useful for unit and integration tests that
need reproducible annotation results without network access.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# ---------------------------------------------------------------------------
# Category response catalogue — one entry per taxonomy v2 category
# ---------------------------------------------------------------------------

_CATEGORY_RESPONSES: dict[str, list[dict[str, Any]]] = {
    "retrieval_failure": [
        {
            "name": "retrieval_failure",
            "confidence": 0.92,
            "evidence": "Agent never searched for the target file despite it existing in the repo.",
        }
    ],
    "query_churn": [
        {
            "name": "query_churn",
            "confidence": 0.85,
            "evidence": "Agent issued 8 distinct search queries with low overlap and no file reads.",
        }
    ],
    "wrong_tool_choice": [
        {
            "name": "wrong_tool_choice",
            "confidence": 0.88,
            "evidence": "Agent used grep 14 times instead of find_references for cross-file analysis.",
        }
    ],
    "over_exploration": [
        {
            "name": "over_exploration",
            "confidence": 0.90,
            "evidence": "Agent made 247 tool calls reading 89 files but only 2 edits. Reward 0.",
        }
    ],
    "edit_verify_loop_failure": [
        {
            "name": "edit_verify_loop_failure",
            "confidence": 0.87,
            "evidence": "Agent edited utils/parser.py five times with alternating errors.",
        }
    ],
    "incomplete_solution": [
        {
            "name": "incomplete_solution",
            "confidence": 0.80,
            "evidence": "Agent fixed quoted fields but missed escaped commas and multiline values.",
        }
    ],
    "near_miss": [
        {
            "name": "near_miss",
            "confidence": 0.83,
            "evidence": "Agent passed 8 of 10 tests; forgot useEffect cleanup function. Reward 0.8.",
        }
    ],
    "minimal_progress": [
        {
            "name": "minimal_progress",
            "confidence": 0.75,
            "evidence": "Agent set up BFS skeleton but left visited-node tracking incomplete.",
        }
    ],
    "rate_limited_run": [
        {
            "name": "rate_limited_run",
            "confidence": 0.95,
            "evidence": "Agent completed in 8 seconds with 0 reward and only 2 tool calls.",
        }
    ],
    "exception_crash": [
        {
            "name": "exception_crash",
            "confidence": 0.97,
            "evidence": "Agent crashed after 12 seconds with ConnectionResetError from MCP server.",
        }
    ],
    "stale_context": [
        {
            "name": "stale_context",
            "confidence": 0.82,
            "evidence": "Agent read file at step 3 but edited at step 15 using stale signature.",
        }
    ],
    "decomposition_failure": [
        {
            "name": "decomposition_failure",
            "confidence": 0.86,
            "evidence": "Agent edited API handler before updating the database schema.",
        }
    ],
    "premature_commit": [
        {
            "name": "premature_commit",
            "confidence": 0.78,
            "evidence": "Agent committed changes before running tests; tests failed post-commit.",
        }
    ],
    "clean_success": [
        {
            "name": "clean_success",
            "confidence": 0.99,
            "evidence": "Agent completed the task efficiently with reward 1.0 and no errors.",
        }
    ],
}

# Deterministic fallback for prompts that don't match any known category
_FALLBACK_RESPONSE: list[dict[str, Any]] = [
    {
        "name": "retrieval_failure",
        "confidence": 0.50,
        "evidence": "Fallback: no specific category detected in prompt.",
    }
]


class FakeLLMBackend:
    """Deterministic fake LLM backend for annotation pipeline testing.

    Returns pre-defined responses keyed by category names found in the input
    prompt.  If no known category is detected, a stable fallback response is
    returned.

    Attributes
    ----------
    call_count : int
        Number of times ``annotate`` has been called.
    call_log : list[str]
        Ordered list of prompt hashes for each call (SHA-256 hex digest).
    """

    def __init__(self) -> None:
        self.call_count: int = 0
        self.call_log: list[str] = []

    def __repr__(self) -> str:
        return f"FakeLLMBackend(call_count={self.call_count})"

    # ----- public API -----

    def annotate(self, prompt: str) -> dict[str, Any]:
        """Return a deterministic annotation response for *prompt*.

        The response is a dict matching the LLM annotation schema::

            {"categories": [{"name": "...", "confidence": 0.9, "evidence": "..."}]}

        Determinism is guaranteed: the same prompt always yields the same
        output, and the selection is based on category keywords in the prompt.
        """
        self.call_count += 1
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        self.call_log.append(prompt_hash)

        categories = self._match_categories(prompt)
        return {"categories": categories}

    def annotate_json(self, prompt: str) -> str:
        """Like ``annotate`` but returns a JSON string."""
        return json.dumps(self.annotate(prompt))

    # ----- internal -----

    @staticmethod
    def _match_categories(prompt: str) -> list[dict[str, Any]]:
        """Select response categories by scanning *prompt* for known names.

        Iterates through ``_CATEGORY_RESPONSES`` in deterministic (insertion)
        order.  Returns responses for ALL matching category names found in the
        prompt.  If none match, returns the fallback response.
        """
        prompt_lower = prompt.lower()
        matched: list[dict[str, Any]] = []
        for category_name, response in _CATEGORY_RESPONSES.items():
            if category_name in prompt_lower:
                matched.extend(response)
        return matched if matched else list(_FALLBACK_RESPONSE)

    @staticmethod
    def available_categories() -> list[str]:
        """Return the list of category names this backend can produce."""
        return list(_CATEGORY_RESPONSES.keys())
