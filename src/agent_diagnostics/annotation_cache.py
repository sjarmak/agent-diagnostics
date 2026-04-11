"""Content-hash annotation cache for LLM annotator results.

Caches LLM annotation responses keyed by SHA-256 hash of (prompt + model_id).
This avoids redundant LLM calls when re-running annotation on the same trial
with the same model.

Default cache directory: ``cache/llm_annotations/`` relative to CWD.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("cache/llm_annotations")


def cache_key(prompt_text: str, model_id: str) -> str:
    """Return a deterministic SHA-256 hex digest for (prompt_text, model_id).

    The key is computed from the concatenation of *prompt_text* and *model_id*,
    ensuring that different models produce different cache keys even for the
    same prompt.
    """
    payload = prompt_text + model_id
    return hashlib.sha256(payload.encode()).hexdigest()


def get_cached(cache_dir: Path, key: str) -> list[dict[str, Any]] | None:
    """Read cached annotation categories from ``{cache_dir}/{key}.json``.

    Returns the parsed list of category dicts on cache hit, or ``None``
    on cache miss (file not found, corrupt JSON, or non-list content).
    """
    path = cache_dir / f"{key}.json"
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return data
        logger.warning("Cache file %s does not contain a list; ignoring", path)
        return None
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cache read error for %s: %s", path, exc)
        return None


def put_cached(
    cache_dir: Path,
    key: str,
    categories: list[dict[str, Any]],
) -> None:
    """Write *categories* to ``{cache_dir}/{key}.json``.

    Creates the cache directory tree if it does not exist.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(json.dumps(categories, indent=1))
