"""Content-hash cache for signal extraction (PRD NH-1).

Caches :class:`TrialSignals` dicts keyed by ``sha256(result.json_bytes +
trajectory.json_bytes)`` so that re-running ``extract_all`` on an unchanged
corpus skips the parse-and-derive step entirely.

The cache is a JSONL file where each line is ``{"hash": "<hex>", "signals":
{...}}``.  On load, all rows are read into an in-memory dict; on a cache miss,
the newly-computed signals are appended to the file (write-through).

.. note::
   Cache invalidation is content-driven — changing ``result.json`` or
   ``trajectory.json`` changes the hash and forces re-extraction.  Stale
   entries accumulate until explicitly pruned; a follow-up command could add
   compaction if the cache grows unbounded.

.. warning::
   This cache is **not** safe for concurrent writers from multiple processes.
   Intended for single-user CLI invocations.  For concurrent use, wrap writes
   in an ``fcntl.flock`` lock around :meth:`SignalsCache.put`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "extract-cache.jsonl"


def compute_content_hash(result_bytes: bytes, trajectory_bytes: bytes | None) -> str:
    """Return ``sha256(result_bytes || 0x00 || trajectory_bytes)`` as hex.

    A null byte separates the two payloads so that different byte boundaries
    cannot collide by accident.  When *trajectory_bytes* is ``None`` (e.g. no
    trajectory file on disk) an empty payload is used.
    """
    hasher = hashlib.sha256()
    hasher.update(result_bytes)
    hasher.update(b"\x00")
    hasher.update(trajectory_bytes or b"")
    return hasher.hexdigest()


class SignalsCache:
    """Content-hash-keyed cache of extracted :class:`TrialSignals` dicts.

    Parameters
    ----------
    cache_dir:
        Directory that will hold ``extract-cache.jsonl``.  Created on first
        write if it does not exist.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._path = self.cache_dir / _CACHE_FILENAME
        self._entries: dict[str, dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._load()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.is_file():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning(
                        "extract_cache: skipping malformed line %d in %s",
                        line_no,
                        self._path,
                    )
                    continue
                content_hash = row.get("hash")
                signals = row.get("signals")
                if isinstance(content_hash, str) and isinstance(signals, dict):
                    self._entries[content_hash] = signals

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, content_hash: str) -> dict[str, Any] | None:
        """Return the cached signals for *content_hash* or ``None`` on miss."""
        entry = self._entries.get(content_hash)
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        # Return a shallow copy so callers cannot mutate the cached row.
        return dict(entry)

    def put(self, content_hash: str, signals: dict[str, Any]) -> None:
        """Store *signals* under *content_hash* and append to the cache file."""
        if content_hash in self._entries:
            return  # already cached; write-through is idempotent
        self._entries[content_hash] = dict(signals)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"hash": content_hash, "signals": signals}) + "\n")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": len(self._entries),
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, content_hash: object) -> bool:
        return content_hash in self._entries
