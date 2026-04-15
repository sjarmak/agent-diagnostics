"""Tests for trial_id computation (P1-P7 property tests + real-corpus collision test)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from agent_diagnostics.signals import compute_trial_id, extract_signals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEX_RE = re.compile(r"^[0-9a-f]+$")


def _write_result(trial_dir: Path, data: dict) -> None:
    """Write a result.json into the trial directory."""
    (trial_dir / "result.json").write_text(json.dumps(data))


def _minimal_result(
    *,
    task_name: str = "fix_bug_123",
    config_name: str = "default",
    model: str = "anthropic/claude-sonnet-4-6",
    started_at: str = "2026-01-01T00:00:00Z",
    finished_at: str = "2026-01-01T00:05:00Z",
    reward: float = 1.0,
) -> dict:
    """Build a minimal valid result.json payload."""
    return {
        "task_name": task_name,
        "config_name": config_name,
        "verifier_result": {"rewards": {"reward": reward}},
        "started_at": started_at,
        "finished_at": finished_at,
        "agent_info": {
            "name": "ClaudeCode",
            "model_info": {"name": model},
        },
    }


# ---------------------------------------------------------------------------
# P1: Same trial, different path -> same trial_id
# ---------------------------------------------------------------------------


def test_p1_same_trial_different_path(tmp_path: Path) -> None:
    """Two directories with identical result.json content produce the same trial_id."""
    result = _minimal_result()

    dir_a = tmp_path / "run_a" / "trial_01"
    dir_a.mkdir(parents=True)
    _write_result(dir_a, result)

    dir_b = tmp_path / "run_b" / "trial_01"
    dir_b.mkdir(parents=True)
    _write_result(dir_b, result)

    sig_a = extract_signals(dir_a, suite_mapping={"fix_bug_123": "test"})
    sig_b = extract_signals(dir_b, suite_mapping={"fix_bug_123": "test"})

    assert sig_a["trial_id"] == sig_b["trial_id"]
    assert sig_a["trial_id_full"] == sig_b["trial_id_full"]


# ---------------------------------------------------------------------------
# P2: Re-scoring (different reward) -> same trial_id
# ---------------------------------------------------------------------------


def test_p2_rescoring_same_trial_id(tmp_path: Path) -> None:
    """Changing the reward does not change the trial_id."""
    dir_a = tmp_path / "trial_reward_1"
    dir_a.mkdir()
    _write_result(dir_a, _minimal_result(reward=1.0))

    dir_b = tmp_path / "trial_reward_0"
    dir_b.mkdir()
    _write_result(dir_b, _minimal_result(reward=0.0))

    sig_a = extract_signals(dir_a, suite_mapping={"fix_bug_123": "test"})
    sig_b = extract_signals(dir_b, suite_mapping={"fix_bug_123": "test"})

    assert sig_a["trial_id"] == sig_b["trial_id"]
    assert sig_a["trial_id_full"] == sig_b["trial_id_full"]


# ---------------------------------------------------------------------------
# P3: Retried trial (different started_at) -> different trial_id
# ---------------------------------------------------------------------------


def test_p3_retried_trial_different_id(tmp_path: Path) -> None:
    """Different started_at values produce different trial_ids."""
    dir_a = tmp_path / "trial_run_1"
    dir_a.mkdir()
    _write_result(dir_a, _minimal_result(started_at="2026-01-01T00:00:00Z"))

    dir_b = tmp_path / "trial_run_2"
    dir_b.mkdir()
    _write_result(dir_b, _minimal_result(started_at="2026-01-02T00:00:00Z"))

    sig_a = extract_signals(dir_a, suite_mapping={"fix_bug_123": "test"})
    sig_b = extract_signals(dir_b, suite_mapping={"fix_bug_123": "test"})

    assert sig_a["trial_id"] != sig_b["trial_id"]
    assert sig_a["trial_id_full"] != sig_b["trial_id_full"]


# ---------------------------------------------------------------------------
# P4: Different task_id or model -> different trial_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,val_a,val_b",
    [
        ("task_name", "task_alpha", "task_beta"),
        ("model", "anthropic/claude-sonnet-4-6", "anthropic/claude-opus-4-5"),
    ],
    ids=["different_task_id", "different_model"],
)
def test_p4_different_key_field_different_id(
    tmp_path: Path,
    field: str,
    val_a: str,
    val_b: str,
) -> None:
    """Different task_id or model values produce different trial_ids."""
    kwargs_a = {("task_name" if field == "task_name" else field): val_a}
    kwargs_b = {("task_name" if field == "task_name" else field): val_b}

    dir_a = tmp_path / "trial_a"
    dir_a.mkdir()
    _write_result(dir_a, _minimal_result(**kwargs_a))

    dir_b = tmp_path / "trial_b"
    dir_b.mkdir()
    _write_result(dir_b, _minimal_result(**kwargs_b))

    sig_a = extract_signals(dir_a, suite_mapping={val_a: "test", val_b: "test"})
    sig_b = extract_signals(dir_b, suite_mapping={val_a: "test", val_b: "test"})

    assert sig_a["trial_id"] != sig_b["trial_id"]
    assert sig_a["trial_id_full"] != sig_b["trial_id_full"]


# ---------------------------------------------------------------------------
# P5: Deterministic — 100 runs same input -> 1 unique id
# ---------------------------------------------------------------------------


def test_p5_deterministic() -> None:
    """compute_trial_id is deterministic across 100 invocations."""
    ids = {
        compute_trial_id("task_1", "config_a", "2026-01-01T00:00:00Z", "model_x")
        for _ in range(100)
    }
    assert len(ids) == 1


# ---------------------------------------------------------------------------
# P6: trial_id is 32 hex, trial_id_full is 64 hex
# ---------------------------------------------------------------------------


def test_p6_hex_lengths() -> None:
    """trial_id is 32 hex chars, trial_id_full is 64 hex chars."""
    trial_id, trial_id_full = compute_trial_id(
        "task_1", "config_a", "2026-01-01T00:00:00Z", "model_x"
    )
    assert len(trial_id) == 32
    assert _HEX_RE.match(trial_id)
    assert len(trial_id_full) == 64
    assert _HEX_RE.match(trial_id_full)


# ---------------------------------------------------------------------------
# P7: trial_id == trial_id_full[:32]
# ---------------------------------------------------------------------------


def test_p7_prefix_relationship() -> None:
    """trial_id is the first 32 characters of trial_id_full."""
    trial_id, trial_id_full = compute_trial_id(
        "task_1", "config_a", "2026-01-01T00:00:00Z", "model_x"
    )
    assert trial_id == trial_id_full[:32]


# ---------------------------------------------------------------------------
# Verify compute_trial_id matches the spec formula exactly
# ---------------------------------------------------------------------------


def test_formula_matches_spec() -> None:
    """Verify the hash matches sha256(task_id||config_name||started_at||model)."""
    task_id = "django__django-12345"
    config_name = "default"
    started_at = "2026-03-15T10:30:00Z"
    model = "anthropic/claude-sonnet-4-6"

    payload = f"{task_id}||{config_name}||{started_at}||{model}"
    expected_full = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    expected_short = expected_full[:32]

    trial_id, trial_id_full = compute_trial_id(task_id, config_name, started_at, model)
    assert trial_id_full == expected_full
    assert trial_id == expected_short


# ---------------------------------------------------------------------------
# Real-corpus collision test (skipped if data/signals.jsonl missing)
# ---------------------------------------------------------------------------

_SIGNALS_JSONL = Path(__file__).resolve().parent.parent / "data" / "signals.jsonl"


@pytest.mark.skipif(
    not _SIGNALS_JSONL.is_file(),
    reason="data/signals.jsonl not found — skipping real-corpus collision test",
)
def test_real_corpus_no_collisions() -> None:
    """No truncated trial_id collisions across the real signals corpus.

    Reads every row from data/signals.jsonl, computes trial_id from
    (task_id, config_name, started_at, model) if available, and asserts
    that no two distinct trial_id_full values share the same 32-char
    trial_id prefix.
    """
    rows: list[dict] = []
    with open(_SIGNALS_JSONL) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))

    # Build mapping: trial_id -> set of trial_id_full
    id_map: dict[str, set[str]] = {}
    for row in rows:
        task_id = row.get("task_id", "")
        config_name = row.get("config_name", "")
        # started_at may not be in older signals — skip rows without it
        started_at = row.get("started_at", "")
        model = row.get("model", "")

        # Recompute from the four fields (started_at not stored yet in
        # old corpus rows; we use whatever is available)
        short, full = compute_trial_id(task_id, config_name, started_at, model)
        id_map.setdefault(short, set()).add(full)

    collisions = {k: v for k, v in id_map.items() if len(v) > 1}
    assert collisions == {}, (
        f"Found {len(collisions)} truncated trial_id collision(s): "
        f"{dict(list(collisions.items())[:5])}"
    )
