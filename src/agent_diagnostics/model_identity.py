"""Resolve between logical annotator identities and Anthropic snapshot model IDs.

Logical identities (e.g. ``llm:haiku-4``) are stable labels used in annotations
and datasets.  Snapshot IDs (e.g. ``claude-haiku-4-5-20251001``) are the concrete
model versions returned by the Anthropic API.

Usage::

    from agent_diagnostics.model_identity import resolve_identity, resolve_snapshot

    resolve_identity("claude-haiku-4-5-20251001")   # -> "llm:haiku-4"
    resolve_snapshot("llm:haiku-4")                  # -> "claude-haiku-4-5-20251001"
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

_MODELS_PATH = Path(__file__).parent / "models.yaml"


@functools.lru_cache(maxsize=1)
def load_models_config() -> dict[str, Any]:
    """Load and return the parsed models.yaml configuration.

    The result is cached so that repeated calls avoid re-reading from disk.
    """
    with _MODELS_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "models" not in data:
        raise ValueError(
            f"Invalid models.yaml: expected top-level 'models' key in {_MODELS_PATH}"
        )

    return data


def _build_snapshot_to_identity_map() -> dict[str, str]:
    """Build a reverse lookup from snapshot ID -> logical identity."""
    config = load_models_config()
    mapping: dict[str, str] = {}
    for logical_id, entry in config["models"].items():
        for snapshot_id in entry.get("snapshot_ids", []):
            mapping[snapshot_id] = logical_id
    return mapping


def resolve_identity(snapshot_id: str) -> str:
    """Return the logical identity for a given Anthropic snapshot model ID.

    Parameters
    ----------
    snapshot_id:
        An Anthropic snapshot model ID such as ``"claude-haiku-4-5-20251001"``.

    Returns
    -------
    str
        The logical identity, e.g. ``"llm:haiku-4"``.

    Raises
    ------
    ValueError
        If *snapshot_id* is not found in ``models.yaml``.
    """
    mapping = _build_snapshot_to_identity_map()
    try:
        return mapping[snapshot_id]
    except KeyError:
        known = ", ".join(sorted(mapping.keys()))
        raise ValueError(
            f"Unknown snapshot ID {snapshot_id!r}. Known snapshot IDs: {known}"
        ) from None


def resolve_snapshot(logical_id: str) -> str:
    """Return the current (first) snapshot ID for a given logical identity.

    Parameters
    ----------
    logical_id:
        A logical annotator identity such as ``"llm:haiku-4"``.

    Returns
    -------
    str
        The first snapshot ID listed for that identity in ``models.yaml``.

    Raises
    ------
    ValueError
        If *logical_id* is not found in ``models.yaml``.
    """
    config = load_models_config()
    entry = config["models"].get(logical_id)
    if entry is None:
        known = ", ".join(sorted(config["models"].keys()))
        raise ValueError(
            f"Unknown logical identity {logical_id!r}. Known identities: {known}"
        ) from None

    snapshot_ids = entry.get("snapshot_ids", [])
    if not snapshot_ids:
        raise ValueError(
            f"Logical identity {logical_id!r} has no snapshot IDs configured."
        )

    return snapshot_ids[0]
