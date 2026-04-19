"""LLM-assisted annotator for the Agent Reliability Observatory.

Reads a trial's task prompt, truncated trajectory, and signal vector,
then asks a Claude model to propose taxonomy categories.  Designed to
calibrate against heuristic annotations and surface categories the
heuristic rules cannot detect (e.g. decomposition_failure, stale_context).

Supports three backends:
- ``claude-code``: Uses the ``claude`` CLI in print mode (default).
  Requires the ``claude`` CLI to be installed and authenticated.
- ``api``: Uses the Anthropic Python SDK directly.
  Requires ``ANTHROPIC_API_KEY`` in the environment.
- ``batch``: Uses the Anthropic Message Batches API for 50% cost
  reduction and no per-minute rate limits.  Submits all requests as
  a server-side batch, polls until complete, then parses results.
  Requires ``ANTHROPIC_API_KEY`` in the environment.

This module is a *package* that re-exports the full previous public API
of the original single-file ``llm_annotator.py``.  Test suites mock
symbols on this package namespace (e.g.
``agent_diagnostics.llm_annotator.get_cached``,
``agent_diagnostics.llm_annotator.shutil.which``), so those names must
remain attributes of this module.  Backend submodules look these helpers
up via ``from agent_diagnostics import llm_annotator as _pkg`` at call
time to honour those patches.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-export stdlib modules that tests patch via this namespace
# (e.g. ``mock.patch("agent_diagnostics.llm_annotator.shutil.which", ...)``).
# ---------------------------------------------------------------------------
import logging  # noqa: F401 — preserved for backward-compatible patching surface
import shutil  # noqa: F401 — patched by test_llm_annotator (``.shutil.which``)
import subprocess  # noqa: F401 — patched by test_llm_annotator (``.subprocess.run``)
import sys  # noqa: F401 — preserved for backward-compatible patching surface

# ---------------------------------------------------------------------------
# Re-export cache helpers so tests mocking
# ``agent_diagnostics.llm_annotator.get_cached`` / ``put_cached`` affect
# runtime backend call sites (which look them up via this package).
# ---------------------------------------------------------------------------
from agent_diagnostics.annotation_cache import (
    DEFAULT_CACHE_DIR,
    cache_key,
    get_cached,
    put_cached,
)

# ---------------------------------------------------------------------------
# Public API and private helpers re-exported from submodules.
# ---------------------------------------------------------------------------
from agent_diagnostics.llm_annotator.api_backend import (
    annotate_batch_api,
    annotate_trial_api,
)
from agent_diagnostics.llm_annotator.batch import annotate_batch_messages
from agent_diagnostics.llm_annotator.claude_code import (
    _annotate_one_claude_code,
    _find_claude_cli,
    annotate_batch_claude_code,
    annotate_trial_claude_code,
)
from agent_diagnostics.llm_annotator.dispatch import annotate_batch, annotate_trial_llm
from agent_diagnostics.llm_annotator.models import (
    _API_MODEL_MAP,
    _MODEL_ALIASES,
    _resolve_model_alias,
    _resolve_model_api,
)
from agent_diagnostics.llm_annotator.parsing import (
    _is_error,
    _load_json,
    _parse_claude_response,
    _read_text,
    _to_annotation_result,
    _unwrap_result,
)
from agent_diagnostics.llm_annotator.prompt import (
    _ANNOTATION_SCHEMA,
    _taxonomy_yaml,
    build_prompt,
    summarise_step,
    truncate_trajectory,
    validate_categories,
)

# ---------------------------------------------------------------------------
# Re-export taxonomy helper so tests mocking
# ``agent_diagnostics.llm_annotator.valid_category_names`` affect
# :func:`validate_categories` at runtime.
# ---------------------------------------------------------------------------
from agent_diagnostics.taxonomy import valid_category_names

# Module logger (kept for backward-compat; submodules define their own).
logger = logging.getLogger(__name__)

__all__ = [
    # model resolution
    "_API_MODEL_MAP",
    "_MODEL_ALIASES",
    "_resolve_model_alias",
    "_resolve_model_api",
    # parsing helpers
    "_is_error",
    "_load_json",
    "_parse_claude_response",
    "_read_text",
    "_to_annotation_result",
    "_unwrap_result",
    # prompt helpers
    "_ANNOTATION_SCHEMA",
    "_taxonomy_yaml",
    "build_prompt",
    "summarise_step",
    "truncate_trajectory",
    "validate_categories",
    # claude-code backend
    "_annotate_one_claude_code",
    "_find_claude_cli",
    "annotate_batch_claude_code",
    "annotate_trial_claude_code",
    # api backend
    "annotate_batch_api",
    "annotate_trial_api",
    # batch backend
    "annotate_batch_messages",
    # dispatch
    "annotate_batch",
    "annotate_trial_llm",
    # re-exported helpers (patched by tests)
    "DEFAULT_CACHE_DIR",
    "cache_key",
    "get_cached",
    "put_cached",
    "valid_category_names",
]
