"""Tests for the content-hash extraction cache (PRD NH-1)."""

from __future__ import annotations

import json
import time
from pathlib import Path


from agent_diagnostics.extract_cache import SignalsCache, compute_content_hash
from agent_diagnostics.signals import extract_all


def _write_trial(root: Path, task_id: str, reward: float = 1.0) -> Path:
    trial_dir = root / f"trial_{task_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "reward": reward,
                "started_at": "2026-04-18T00:00:00Z",
                "agent_info": {"name": "claude-code"},
                "config_name": "baseline",
            }
        )
    )
    (trial_dir / "trajectory.json").write_text(
        json.dumps({"steps": [{"tool": "Read", "result": "ok"}]})
    )
    return trial_dir


class TestContentHash:
    def test_deterministic(self) -> None:
        a = compute_content_hash(b"hello", b"world")
        b = compute_content_hash(b"hello", b"world")
        assert a == b

    def test_changes_with_result_bytes(self) -> None:
        a = compute_content_hash(b"hello", b"world")
        b = compute_content_hash(b"hell0", b"world")
        assert a != b

    def test_changes_with_trajectory_bytes(self) -> None:
        a = compute_content_hash(b"hello", b"world")
        b = compute_content_hash(b"hello", b"w0rld")
        assert a != b

    def test_null_trajectory_is_empty(self) -> None:
        a = compute_content_hash(b"hello", None)
        b = compute_content_hash(b"hello", b"")
        assert a == b

    def test_boundary_distinction(self) -> None:
        """Null separator prevents collision between (a+b) and (ab+'')."""
        a = compute_content_hash(b"foo", b"bar")
        b = compute_content_hash(b"foobar", None)
        assert a != b


class TestSignalsCache:
    def test_miss_then_hit(self, tmp_path: Path) -> None:
        cache = SignalsCache(tmp_path)
        assert cache.get("abc") is None
        cache.put("abc", {"trial_id": "t1"})
        assert cache.get("abc") == {"trial_id": "t1"}
        assert cache.stats == {"hits": 1, "misses": 1, "entries": 1}

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        cache1 = SignalsCache(tmp_path)
        cache1.put("abc", {"trial_id": "t1"})

        cache2 = SignalsCache(tmp_path)
        assert cache2.get("abc") == {"trial_id": "t1"}

    def test_put_idempotent(self, tmp_path: Path) -> None:
        cache = SignalsCache(tmp_path)
        cache.put("abc", {"trial_id": "t1"})
        cache.put("abc", {"trial_id": "t1"})  # should not duplicate line
        lines = cache.path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_returns_copy_to_prevent_mutation(self, tmp_path: Path) -> None:
        cache = SignalsCache(tmp_path)
        cache.put("abc", {"trial_id": "t1"})
        got = cache.get("abc")
        assert got is not None
        got["trial_id"] = "MUTATED"
        fresh = cache.get("abc")
        assert fresh == {"trial_id": "t1"}

    def test_malformed_line_is_skipped(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "extract-cache.jsonl"
        cache_file.write_text(
            '{"hash": "abc", "signals": {"x": 1}}\n'
            "NOT JSON\n"
            '{"hash": "def", "signals": {"y": 2}}\n'
        )
        cache = SignalsCache(tmp_path)
        assert cache.get("abc") == {"x": 1}
        assert cache.get("def") == {"y": 2}
        assert len(cache) == 2


class TestExtractAllWithCache:
    def test_cold_run_populates_cache(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        _write_trial(runs, "t1")
        _write_trial(runs, "t2")

        cache = SignalsCache(tmp_path / "cache")
        signals = extract_all(runs, cache=cache, suite_mapping={"t": "testbench"})

        assert len(signals) == 2
        assert cache.stats["misses"] == 2
        assert cache.stats["hits"] == 0
        assert len(cache) == 2

    def test_warm_run_hits_cache_and_returns_identical_signals(
        self, tmp_path: Path
    ) -> None:
        runs = tmp_path / "runs"
        _write_trial(runs, "t1")
        _write_trial(runs, "t2")

        cache_dir = tmp_path / "cache"

        cold_cache = SignalsCache(cache_dir)
        cold_signals = extract_all(runs, cache=cold_cache)

        warm_cache = SignalsCache(cache_dir)
        warm_signals = extract_all(runs, cache=warm_cache)

        assert warm_cache.stats["hits"] == 2
        assert warm_cache.stats["misses"] == 0

        def _key(s: dict) -> str:
            return str(s.get("task_id", ""))

        assert sorted(cold_signals, key=_key) == sorted(warm_signals, key=_key)

    def test_content_change_invalidates_entry(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        _write_trial(runs, "t1")

        cache = SignalsCache(tmp_path / "cache")
        extract_all(runs, cache=cache)
        assert cache.stats["misses"] == 1

        # Modify trajectory content
        (runs / "trial_t1" / "trajectory.json").write_text(
            json.dumps({"steps": [{"tool": "Edit", "result": "changed"}]})
        )

        extract_all(runs, cache=cache)
        # Original entry still there, plus one new entry for the modified content
        assert cache.stats["misses"] == 2
        assert len(cache) == 2

    def test_warm_run_under_acceptance_threshold(self, tmp_path: Path) -> None:
        """NH-1 acceptance: warm re-extraction of 50 trials is fast.

        Bead spec: warm re-extraction of the full corpus under 5 seconds.
        Scaled-down version (50 trials) for fast test execution.
        """
        runs = tmp_path / "runs"
        for i in range(50):
            _write_trial(runs, f"t{i}")

        cache_dir = tmp_path / "cache"
        cold_cache = SignalsCache(cache_dir)
        extract_all(runs, cache=cold_cache)

        warm_cache = SignalsCache(cache_dir)
        start = time.perf_counter()
        warm_signals = extract_all(runs, cache=warm_cache)
        elapsed = time.perf_counter() - start

        assert len(warm_signals) == 50
        assert warm_cache.stats["hits"] == 50
        # Generous bound — on a dev laptop this is typically <100ms
        assert elapsed < 5.0, f"warm extraction took {elapsed:.2f}s, expected <5s"

    def test_no_cache_argument_preserves_existing_behavior(
        self, tmp_path: Path
    ) -> None:
        runs = tmp_path / "runs"
        _write_trial(runs, "t1")
        signals = extract_all(runs)
        assert len(signals) == 1
        assert signals[0]["task_id"] == "t1"


class TestCacheCliIntegration:
    def test_cmd_extract_with_cache_dir(self, tmp_path: Path) -> None:
        """``observatory extract --cache-dir ...`` populates the cache file."""
        import argparse

        from agent_diagnostics.cli import cmd_extract

        runs = tmp_path / "runs"
        _write_trial(runs, "t1")
        _write_trial(runs, "t2")

        cache_dir = tmp_path / "cache"
        output = tmp_path / "signals.jsonl"
        args = argparse.Namespace(
            runs_dir=str(runs),
            output=str(output),
            cache_dir=str(cache_dir),
        )
        cmd_extract(args)

        assert output.exists()
        assert (cache_dir / "extract-cache.jsonl").is_file()
        lines = (cache_dir / "extract-cache.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
