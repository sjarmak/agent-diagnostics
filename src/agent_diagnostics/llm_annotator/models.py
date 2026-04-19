"""Model-alias / model-ID resolution for the LLM annotator backends."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Model alias mapping (used by both backends)
# ---------------------------------------------------------------------------

_MODEL_ALIASES = {
    "haiku": "haiku",
    "sonnet": "sonnet",
    "opus": "opus",
}

_API_MODEL_MAP = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
}


def _resolve_model_alias(model: str) -> str:
    """Resolve to a short alias for claude CLI (haiku/sonnet/opus)."""
    return _MODEL_ALIASES.get(model, model)


def _resolve_model_api(model: str) -> str:
    """Resolve to a full model ID for the Anthropic API."""
    return _API_MODEL_MAP.get(model, model)
