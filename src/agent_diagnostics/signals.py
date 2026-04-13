"""Signal extraction from trial directories.

Reads result.json and trajectory.json from a trial directory and produces
a :class:`TrialSignals` dict.  This module is the generic, injectable
replacement for CSB's hardcoded signals.py.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_diagnostics.constants import REDACTED_SIGNAL_FIELDS  # noqa: F401
from agent_diagnostics.tool_registry import (
    DEFAULT_REGISTRY,
    ToolRegistry,
    get_registry_for_agent,
)
from agent_diagnostics.types import TrialSignals

# Sentinel object to detect when tool_registry was not explicitly provided.
_REGISTRY_NOT_SET: ToolRegistry = ToolRegistry(
    search_tools=frozenset(),
    edit_tools=frozenset(),
    code_nav_tools=frozenset(),
    semantic_search_tools=frozenset(),
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_EXCLUDED_DIR_PATTERNS: tuple[str, ...] = (
    "__archived_invalid",
    "__incomplete",
    "__pre_sgenv_fix",
    "__verifier_path_bug",
    "__doubled_prefix",
)


def _is_valid_trial(data: dict[str, Any]) -> bool:
    """Check whether *data* (from result.json) represents a valid trial.

    Returns ``False`` for harness summaries and other non-trial records
    that lack the ``agent_info`` key.
    """
    return "agent_info" in data


def _is_excluded_path(trial_dir: Path) -> bool:
    """Return ``True`` if any path component matches an excluded pattern."""
    parts = trial_dir.parts
    return any(pattern in parts for pattern in _EXCLUDED_DIR_PATTERNS)


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning ``None`` if missing or malformed."""
    try:
        with open(path) as f:
            return json.load(f)  # type: ignore[no-any-return]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _extract_reward(data: dict[str, Any]) -> float | None:
    """Extract reward from result.json's ``verifier_result.rewards``."""
    verifier = data.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    reward = rewards.get("reward")
    if reward is None:
        reward = rewards.get("score")
    if reward is not None:
        return float(reward)
    return None


def _extract_duration(result: dict[str, Any] | None) -> float | None:
    """Derive wall-clock seconds from result.json timestamps."""
    if result is None:
        return None
    started = result.get("started_at")
    finished = result.get("finished_at")
    if not started or not finished:
        return None
    try:
        t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        return (t1 - t0).total_seconds()
    except (ValueError, TypeError):
        return None


def _get_nested(data: dict[str, Any] | None, *keys: str) -> Any:
    """Safely traverse nested dicts, returning ``None`` on any miss."""
    val: Any = data
    for k in keys:
        if not isinstance(val, dict):
            return None
        val = val.get(k)
    return val


# ---------------------------------------------------------------------------
# File-path extraction from tool arguments
# ---------------------------------------------------------------------------

_FILE_PATH_KEYS = ("file_path", "path")


def _extract_file_path(tc: dict[str, Any]) -> str | None:
    """Return the file path argument from a tool call, if present."""
    args = tc.get("arguments") or {}
    if isinstance(args, str):
        return None
    for key in _FILE_PATH_KEYS:
        val = args.get(key)
        if val and isinstance(val, str):
            return val
    return None


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------


def _parse_trajectory(
    traj: dict[str, Any] | None,
    registry: ToolRegistry,
) -> dict[str, Any]:
    """Parse trajectory.json and compute signal values.

    Returns a dict with keys matching a subset of :class:`TrialSignals`.
    """
    empty: dict[str, Any] = {
        "total_turns": 0,
        "tool_calls_total": 0,
        "search_tool_calls": 0,
        "edit_tool_calls": 0,
        "code_nav_tool_calls": 0,
        "semantic_search_tool_calls": 0,
        "unique_files_read": 0,
        "unique_files_edited": 0,
        "files_read_list": [],
        "files_edited_list": [],
        "error_count": 0,
        "retry_count": 0,
        "trajectory_length": 0,
        "patch_size_lines": 0,
        "tool_call_sequence": [],
    }
    if traj is None:
        return empty

    steps = traj.get("steps") or []
    if not steps:
        return empty

    total_turns = len(steps)
    tool_call_sequence: list[str] = []
    search_count = 0
    edit_count = 0
    code_nav_count = 0
    semantic_count = 0
    error_count = 0
    retry_count = 0
    patch_size_lines = 0
    files_read: set[str] = set()
    files_edited: set[str] = set()

    # For retry detection: track (function_name, args_key) of previous call
    prev_call_key: tuple[str, str] | None = None

    # Read-capable tool names: Read + any code_nav tool that reads files
    _read_tools = (
        frozenset({"Read", "read_file", "mcp__sourcegraph__read_file"})
        | registry.code_nav_tools
    )
    _edit_tools_for_files = (
        frozenset({"Edit", "Write", "file_write"}) | registry.edit_tools
    )

    for step in steps:
        step_has_error = False
        obs = step.get("observation") or {}

        # Check observation for errors
        obs_str = ""
        if isinstance(obs, dict):
            results = obs.get("results") or []
            for r in results:
                content = str(r.get("content", ""))
                obs_str += content.lower()
        elif isinstance(obs, str):
            obs_str = obs.lower()

        if any(kw in obs_str for kw in ("error", "fail", "exception", "traceback")):
            step_has_error = True

        for tc in step.get("tool_calls") or []:
            fn = tc.get("function_name", "")
            tool_call_sequence.append(fn)

            # Categorize tool
            if fn in registry.search_tools:
                search_count += 1
            if fn in registry.edit_tools:
                edit_count += 1
            if fn in registry.code_nav_tools:
                code_nav_count += 1
            if fn in registry.semantic_search_tools:
                semantic_count += 1

            # File path extraction
            fpath = _extract_file_path(tc)
            if fpath:
                if fn in _read_tools:
                    files_read.add(fpath)
                if fn in _edit_tools_for_files:
                    files_edited.add(fpath)

            # Patch size: count lines in edit/write arguments
            args = tc.get("arguments") or {}
            if isinstance(args, dict) and fn in _edit_tools_for_files:
                for content_key in ("new_string", "content", "new_content"):
                    content = args.get(content_key, "")
                    if isinstance(content, str) and content:
                        patch_size_lines += content.count("\n") + 1

            # Retry detection: same function + same args as previous call
            args_key = (
                json.dumps(args, sort_keys=True)
                if isinstance(args, dict)
                else str(args)
            )
            call_key = (fn, args_key)
            if call_key == prev_call_key:
                retry_count += 1
            prev_call_key = call_key

        if step_has_error:
            error_count += 1

    return {
        "total_turns": total_turns,
        "tool_calls_total": len(tool_call_sequence),
        "search_tool_calls": search_count,
        "edit_tool_calls": edit_count,
        "code_nav_tool_calls": code_nav_count,
        "semantic_search_tool_calls": semantic_count,
        "unique_files_read": len(files_read),
        "unique_files_edited": len(files_edited),
        "files_read_list": sorted(files_read),
        "files_edited_list": sorted(files_edited),
        "error_count": error_count,
        "retry_count": retry_count,
        "trajectory_length": len(tool_call_sequence),
        "patch_size_lines": patch_size_lines,
        "tool_call_sequence": tool_call_sequence,
    }


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_KEYWORDS: dict[str, str] = {
    "opus": "anthropic/claude-opus-4-5",
    "sonnet": "anthropic/claude-sonnet-4-6",
    "haiku": "anthropic/claude-haiku-4-5",
}


def _resolve_model(
    result: dict[str, Any] | None,
    trial_dir: Path,
    model_keywords: dict[str, str] | None,
) -> str | None:
    """Resolve model name from result.json metadata or directory name."""
    # Primary: result.json agent_info.model_info.name
    model = _get_nested(result, "agent_info", "model_info", "name")
    if model:
        return str(model)

    # Secondary: result.json agent_result.model
    model = _get_nested(result, "agent_result", "model")
    if model:
        return str(model)

    # Tertiary: match directory name segments against model_keywords
    if model_keywords:
        dir_name = trial_dir.name.lower()
        # Also check parent names
        for part in (trial_dir.name, *[p for p in trial_dir.parts[-4:]]):
            for keyword, model_name in model_keywords.items():
                if keyword in part.lower().split("_"):
                    return model_name

    return None


# ---------------------------------------------------------------------------
# Benchmark resolution
# ---------------------------------------------------------------------------


# Known directory-name patterns mapping substrings to benchmark names.
_DIRECTORY_BENCHMARK_PATTERNS: dict[str, str] = {
    "crossrepo": "crossrepo",
    "openhands": "openhands",
    "swe-bench": "swe-bench",
    "swe_bench": "swe-bench",
}


def load_manifest(path: Path) -> dict[str, str]:
    """Parse a MANIFEST.json file and return a suite_mapping dict.

    The manifest maps run directory names to benchmark names, e.g.::

        {
            "csb_crossrepo_run1": "crossrepo",
            "openhands_eval_42": "openhands"
        }

    Returns an empty dict if the file is missing or malformed.
    """
    data = _load_json(path)
    if not isinstance(data, dict):
        return {}
    return {
        str(k): str(v)
        for k, v in data.items()
        if isinstance(k, str) and isinstance(v, str)
    }


def _resolve_benchmark_from_directory(trial_dir: Path) -> str | None:
    """Resolve benchmark name from directory-name conventions.

    Checks path components for known benchmark substrings such as
    ``crossrepo``, ``openhands``, and ``swe-bench``/``swe_bench``.
    """
    parts_to_check = (
        trial_dir.parts[-4:] if len(trial_dir.parts) >= 4 else trial_dir.parts
    )
    for part in parts_to_check:
        part_lower = part.lower()
        for pattern, benchmark in _DIRECTORY_BENCHMARK_PATTERNS.items():
            if pattern in part_lower:
                return benchmark
    return None


def _resolve_benchmark(
    task_id: str | None,
    suite_mapping: dict[str, str] | None,
    benchmark_resolver: Callable[[Path], str | None] | None,
    trial_dir: Path,
) -> tuple[str | None, str]:
    """Resolve benchmark name via suite_mapping prefix match or callable.

    Returns a tuple of ``(benchmark_name, source)`` where *source* is one of
    ``"manifest"`` (resolved via suite_mapping or benchmark_resolver),
    ``"directory"`` (resolved via directory-name convention), or ``""``
    (unresolved).
    """
    # Try benchmark_resolver first (most specific)
    if benchmark_resolver is not None:
        result = benchmark_resolver(trial_dir)
        if result:
            return result, "manifest"

    # Try suite_mapping prefix match against task_id
    if suite_mapping and task_id:
        # Sort by descending prefix length for correct longest-match
        for prefix in sorted(suite_mapping, key=len, reverse=True):
            if task_id.startswith(prefix):
                return suite_mapping[prefix], "manifest"

    # Try suite_mapping prefix match against directory name
    if suite_mapping:
        dir_name = trial_dir.name
        for prefix in sorted(suite_mapping, key=len, reverse=True):
            if dir_name.startswith(prefix):
                return suite_mapping[prefix], "manifest"

    # Try directory-name convention fallback
    dir_result = _resolve_benchmark_from_directory(trial_dir)
    if dir_result:
        return dir_result, "directory"

    return None, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_signals(
    trial_dir: Path,
    *,
    tool_registry: ToolRegistry = _REGISTRY_NOT_SET,
    suite_mapping: dict[str, str] | None = None,
    benchmark_resolver: Callable[[Path], str | None] | None = None,
    task_id_normalizer: Callable[[str], str] | None = None,
    model_keywords: dict[str, str] | None = None,
) -> TrialSignals:
    """Extract quantitative reliability signals from a single trial directory.

    Reads ``result.json`` and ``trajectory.json`` from *trial_dir* and
    produces a :class:`TrialSignals` dict with all 26 keys populated
    (or set to sensible defaults when data is missing).

    When *tool_registry* is not explicitly provided, the function reads
    ``result.json``'s ``agent_info.name`` field and auto-selects the
    appropriate registry via :func:`get_registry_for_agent`.

    Parameters
    ----------
    trial_dir:
        Path to a trial directory (should contain ``result.json``).
    tool_registry:
        Tool name classification registry.  When omitted, auto-detected
        from ``agent_info.name`` in ``result.json``; falls back to
        :data:`DEFAULT_REGISTRY`.
    suite_mapping:
        Optional ``{prefix: benchmark_name}`` dict for benchmark
        resolution via prefix matching on ``task_id``.
    benchmark_resolver:
        Optional callable ``(Path) -> str | None`` for custom benchmark
        resolution.
    task_id_normalizer:
        Optional callable ``(str) -> str`` applied to the raw task_id.
    model_keywords:
        Optional ``{keyword: model_name}`` dict for model resolution
        from directory names.
    """
    # Warn when no benchmark resolution is configured
    if suite_mapping is None and benchmark_resolver is None:
        warnings.warn(
            "Running without suite_mapping or benchmark_resolver; "
            "benchmark field will be None, which reduces annotation quality.",
            UserWarning,
            stacklevel=2,
        )

    # --- Load files ---
    result_path = trial_dir / "result.json"
    traj_path = trial_dir / "trajectory.json"
    traj_path_nested = trial_dir / "agent" / "trajectory.json"

    result = _load_json(result_path)
    traj = _load_json(traj_path) or _load_json(traj_path_nested)

    has_result_json = result is not None
    has_trajectory = traj is not None

    # --- Auto-detect tool registry from agent_info.name ---
    if tool_registry is _REGISTRY_NOT_SET:
        agent_name = _get_nested(result, "agent_info", "name")
        if agent_name and isinstance(agent_name, str):
            tool_registry = get_registry_for_agent(agent_name)
        else:
            tool_registry = DEFAULT_REGISTRY

    # --- Reward / passed ---
    reward: float | None = None
    if result:
        reward = _extract_reward(result)
    passed = reward is not None and reward > 0

    # --- Task ID ---
    task_id: str | None = _get_nested(result, "task_name")
    if task_id is None:
        task_id = _get_nested(result, "task_id")
    if task_id and task_id_normalizer:
        task_id = task_id_normalizer(task_id)

    # --- Model ---
    model = _resolve_model(result, trial_dir, model_keywords)

    # --- Config name ---
    config_name: str | None = _get_nested(result, "config_name")
    if config_name is None:
        config_name = _get_nested(result, "config", "name")

    # --- Benchmark ---
    benchmark, benchmark_source = _resolve_benchmark(
        task_id, suite_mapping, benchmark_resolver, trial_dir
    )

    # --- Duration ---
    duration_seconds = _extract_duration(result)

    # --- Exception / rate limit ---
    exception_crashed = False
    rate_limited = False
    if result:
        exc_info = result.get("exception_info")
        if exc_info:
            if isinstance(exc_info, dict) and exc_info:
                exception_crashed = True
                exc_str = json.dumps(exc_info).lower()
                rate_limited = "rate_limit" in exc_str or "rate limit" in exc_str
            elif isinstance(exc_info, str) and exc_info.strip():
                exception_crashed = True
                rate_limited = (
                    "rate_limit" in exc_info.lower() or "rate limit" in exc_info.lower()
                )

    # --- Trajectory parsing ---
    traj_signals = _parse_trajectory(traj, tool_registry)

    # --- Build TrialSignals ---
    signals: TrialSignals = {
        "task_id": task_id or "",
        "model": model or "",
        "config_name": config_name or "",
        "benchmark": benchmark or "",
        "reward": reward if reward is not None else 0.0,
        "passed": passed,
        "total_turns": traj_signals["total_turns"],
        "tool_calls_total": traj_signals["tool_calls_total"],
        "search_tool_calls": traj_signals["search_tool_calls"],
        "edit_tool_calls": traj_signals["edit_tool_calls"],
        "code_nav_tool_calls": traj_signals["code_nav_tool_calls"],
        "semantic_search_tool_calls": traj_signals["semantic_search_tool_calls"],
        "unique_files_read": traj_signals["unique_files_read"],
        "unique_files_edited": traj_signals["unique_files_edited"],
        "files_read_list": traj_signals["files_read_list"],
        "files_edited_list": traj_signals["files_edited_list"],
        "error_count": traj_signals["error_count"],
        "retry_count": traj_signals["retry_count"],
        "trajectory_length": traj_signals["trajectory_length"],
        "has_result_json": has_result_json,
        "has_trajectory": has_trajectory,
        "duration_seconds": duration_seconds if duration_seconds is not None else 0.0,
        "rate_limited": rate_limited,
        "exception_crashed": exception_crashed,
        "patch_size_lines": traj_signals["patch_size_lines"],
        "tool_call_sequence": traj_signals["tool_call_sequence"],
        "benchmark_source": benchmark_source,
    }

    return signals


def judge_safe_signals(signals: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *signals* with reward-leaking fields removed.

    This produces a judge-safe subset suitable for LLM annotation prompts.
    Fields derived from ground-truth evaluation data are stripped so that
    the judge cannot reverse-engineer the correct answer.

    Redacted fields
    ---------------
    - ``reward`` — raw verifier score
    - ``passed`` — binary pass/fail derived from reward
    - ``exception_info`` — raw exception payload from the harness
    - ``exception_crashed`` — derived from ``exception_info``

    Any key present in :data:`REDACTED_SIGNAL_FIELDS` is removed, along with
    the ``exception_crashed`` field which is derived from ``exception_info``.
    """
    derived_fields: frozenset[str] = frozenset({"exception_crashed"})
    exclude = REDACTED_SIGNAL_FIELDS | derived_fields
    return {k: v for k, v in signals.items() if k not in exclude}


def extract_all(
    root_dir: Path,
    *,
    tool_registry: ToolRegistry = _REGISTRY_NOT_SET,
    suite_mapping: dict[str, str] | None = None,
    benchmark_resolver: Callable[[Path], str | None] | None = None,
    task_id_normalizer: Callable[[str], str] | None = None,
    model_keywords: dict[str, str] | None = None,
) -> list[TrialSignals]:
    """Walk a directory tree and extract signals from every trial.

    A trial directory is any directory containing ``result.json``.

    Parameters
    ----------
    root_dir:
        Root directory to walk.
    tool_registry, suite_mapping, benchmark_resolver, task_id_normalizer, model_keywords:
        Forwarded to :func:`extract_signals`.

    Returns
    -------
    list[TrialSignals]
        One signal dict per trial directory found.
    """
    results: list[TrialSignals] = []
    for result_file in sorted(root_dir.rglob("result.json")):
        trial_dir = result_file.parent

        # Skip directories matching excluded patterns
        if _is_excluded_path(trial_dir):
            continue

        # Skip invalid trials (e.g. harness summaries without agent_info)
        data = _load_json(result_file)
        if data is None or not _is_valid_trial(data):
            continue

        signals = extract_signals(
            trial_dir,
            tool_registry=tool_registry,
            suite_mapping=suite_mapping,
            benchmark_resolver=benchmark_resolver,
            task_id_normalizer=task_id_normalizer,
            model_keywords=model_keywords,
        )
        results.append(signals)
    return results
