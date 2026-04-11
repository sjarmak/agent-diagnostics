"""Shared constants for the agent-diagnostics package."""

from __future__ import annotations

# Signal fields that contain ground-truth or sensitive evaluation data
# and must be redacted from LLM annotation prompts to prevent data leakage.
REDACTED_SIGNAL_FIELDS: frozenset[str] = frozenset(
    {"reward", "passed", "exception_info"}
)
