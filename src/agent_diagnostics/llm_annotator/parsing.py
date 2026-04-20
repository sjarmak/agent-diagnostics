"""Trial file loading, response parsing, and AnnotationResult helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_diagnostics.types import (
    AnnotationError,
    AnnotationNoCategoriesFound,
    AnnotationOk,
    AnnotationResult,
)

# ---------------------------------------------------------------------------
# Trial data loading
# ---------------------------------------------------------------------------


def _read_text(path: Path, max_chars: int = 0) -> str | None:
    """Read a text file, optionally truncating to *max_chars*."""
    try:
        text = path.read_text(errors="replace")
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated]"
        return text
    except (FileNotFoundError, OSError):
        return None


def _load_json(path: Path) -> Any | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# AnnotationResult helpers
# ---------------------------------------------------------------------------


def _to_annotation_result(categories: list[dict]) -> AnnotationResult:
    """Wrap a validated category list into the appropriate AnnotationResult variant."""
    if categories:
        return AnnotationOk(categories=tuple(categories))
    return AnnotationNoCategoriesFound()


def _unwrap_result(result: AnnotationResult) -> list[dict]:
    """Return the category list for :class:`AnnotationOk`, else ``[]``."""
    if isinstance(result, AnnotationOk):
        return list(result.categories)
    return []


def _is_error(result: AnnotationResult) -> bool:
    """Return ``True`` if *result* represents a failed annotation."""
    return isinstance(result, AnnotationError)


# Cap on exception-derived reason strings.  Some network exceptions
# (e.g. httpx.HTTPStatusError wrapping a large response body) can carry
# multi-kilobyte ``__str__`` output; truncate before persistence and logging
# so adversarially-large reasons cannot cause memory bloat or leak response
# bodies verbatim into logs.
_MAX_ERROR_REASON_LEN = 500


def _short_exc(exc: BaseException) -> str:
    """Return ``str(exc)`` truncated to :data:`_MAX_ERROR_REASON_LEN`."""
    body = str(exc)
    if len(body) > _MAX_ERROR_REASON_LEN:
        return body[:_MAX_ERROR_REASON_LEN] + "... [truncated]"
    return body


# ---------------------------------------------------------------------------
# Shared response parsing (used by both claude-code backends)
# ---------------------------------------------------------------------------


def _parse_claude_response(envelope: dict) -> list[dict] | None:
    """Parse a Claude CLI JSON envelope and return raw category dicts.

    Returns a list of category dicts on success, or ``None`` when the
    envelope indicates an error, empty result, bad JSON, or non-list
    categories.  The caller is responsible for running
    ``validate_categories`` on the returned list.
    """
    if envelope.get("is_error"):
        return None

    structured = envelope.get("structured_output")
    if structured and isinstance(structured, dict):
        categories = structured.get("categories", [])
    else:
        raw = envelope.get("result", "").strip()
        if not raw:
            return None
        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            raw = "\n".join(lines).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        categories = parsed.get("categories", parsed) if isinstance(parsed, dict) else parsed

    if not isinstance(categories, list):
        return None

    return categories
