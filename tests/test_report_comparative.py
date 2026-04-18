"""Tests for comparative analysis sections in the reliability report.

Covers per-benchmark summary, per-agent summary, agent x benchmark cross-tab
matrix, and top-divergences helpers added for bead `agent-diagnostics-ekb`.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from agent_diagnostics.report import (
    LOW_CONFIDENCE_N,
    _agent_benchmark_matrix,
    _build_trials_frame,
    _per_agent_summary,
    _per_benchmark_summary,
    _top_divergences,
    generate_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk(
    *,
    agent: str | None = "claude-code",
    model: str | None = "sonnet-4.6",
    benchmark: str | None = "swe-bench",
    passed: bool,
    reward: float,
    config_name: str = "baseline",
    categories: list[dict] | None = None,
    task_id: str = "t",
    trajectory_length: int | None = None,
    total_turns: int | None = None,
) -> dict:
    """Build a synthetic annotation dict."""
    ann: dict = {
        "task_id": task_id,
        "config_name": config_name,
        "passed": passed,
        "reward": reward,
        "categories": categories or [],
    }
    if agent is not None:
        ann["agent_name"] = agent
    if model is not None:
        ann["model"] = model
    if benchmark is not None:
        ann["benchmark"] = benchmark
    if trajectory_length is not None:
        ann["trajectory_length"] = trajectory_length
    if total_turns is not None:
        ann["total_turns"] = total_turns
    return ann


@pytest.fixture()
def rich_annotations() -> list[dict]:
    """3 agents x 4 benchmarks x 5 trials per cell, with varied pass rates.

    Pass rates (planned):
      claude-code  x swe-bench   = 5/5 = 1.00
      claude-code  x humaneval   = 3/5 = 0.60
      claude-code  x mbpp        = 2/5 = 0.40
      claude-code  x polyglot    = 1/5 = 0.20
      openhands    x swe-bench   = 2/5 = 0.40
      openhands    x humaneval   = 4/5 = 0.80
      openhands    x mbpp        = 2/5 = 0.40
      openhands    x polyglot    = 0/5 = 0.00
      cursor-cli   x swe-bench   = 3/5 = 0.60
      cursor-cli   x humaneval   = 3/5 = 0.60
      cursor-cli   x mbpp        = 3/5 = 0.60
      cursor-cli   x polyglot    = 3/5 = 0.60
    """
    plan = {
        ("claude-code", "swe-bench"): 5,
        ("claude-code", "humaneval"): 3,
        ("claude-code", "mbpp"): 2,
        ("claude-code", "polyglot"): 1,
        ("openhands", "swe-bench"): 2,
        ("openhands", "humaneval"): 4,
        ("openhands", "mbpp"): 2,
        ("openhands", "polyglot"): 0,
        ("cursor-cli", "swe-bench"): 3,
        ("cursor-cli", "humaneval"): 3,
        ("cursor-cli", "mbpp"): 3,
        ("cursor-cli", "polyglot"): 3,
    }
    models = {
        "claude-code": "sonnet-4.6",
        "openhands": "gpt-4",
        "cursor-cli": "sonnet-4.6",
    }
    out: list[dict] = []
    tid = 0
    for (agent, bench), passed_count in plan.items():
        for i in range(5):
            tid += 1
            is_pass = i < passed_count
            cats: list[dict] = []
            if not is_pass:
                cats.append(
                    {
                        "name": "retrieval_failure",
                        "confidence": 0.8,
                        "evidence": "x",
                    }
                )
            out.append(
                _mk(
                    agent=agent,
                    model=models[agent],
                    benchmark=bench,
                    passed=is_pass,
                    reward=1.0 if is_pass else 0.0,
                    categories=cats,
                    task_id=f"task_{tid:03d}",
                    trajectory_length=10,
                    total_turns=3,
                )
            )
    return out


# ---------------------------------------------------------------------------
# _build_trials_frame
# ---------------------------------------------------------------------------


class TestBuildTrialsFrame:
    def test_missing_agent_name_becomes_unknown(self) -> None:
        anns = [_mk(agent=None, passed=True, reward=1.0)]
        frame = _build_trials_frame(anns)
        assert len(frame) == 1
        assert frame[0]["agent"] == "unknown"

    def test_missing_benchmark_becomes_unknown(self) -> None:
        anns = [_mk(benchmark=None, passed=True, reward=1.0)]
        frame = _build_trials_frame(anns)
        assert frame[0]["benchmark"] == "unknown"

    def test_empty_string_fields_become_unknown(self) -> None:
        anns = [_mk(agent="", benchmark="", passed=False, reward=0.0)]
        frame = _build_trials_frame(anns)
        assert frame[0]["agent"] == "unknown"
        assert frame[0]["benchmark"] == "unknown"

    def test_does_not_mutate_input(self) -> None:
        anns = [_mk(agent=None, benchmark=None, passed=True, reward=0.5)]
        snapshot = deepcopy(anns)
        _build_trials_frame(anns)
        assert anns == snapshot

    def test_carries_passed_reward_and_categories(self) -> None:
        cats = [{"name": "retrieval_failure", "confidence": 0.8, "evidence": ""}]
        anns = [_mk(passed=False, reward=0.25, categories=cats)]
        frame = _build_trials_frame(anns)
        row = frame[0]
        assert row["passed"] is False
        assert row["reward"] == 0.25
        assert row["categories"] == cats

    def test_carries_model_with_none_when_missing(self) -> None:
        anns = [_mk(model=None, passed=True, reward=1.0)]
        frame = _build_trials_frame(anns)
        assert frame[0]["model"] is None


# ---------------------------------------------------------------------------
# _per_benchmark_summary
# ---------------------------------------------------------------------------


class TestPerBenchmarkSummary:
    def test_aggregates_by_benchmark(self, rich_annotations: list[dict]) -> None:
        frame = _build_trials_frame(rich_annotations)
        result = _per_benchmark_summary(frame)

        by_name = {r["benchmark"]: r for r in result}
        # Each benchmark has 15 trials (5 per agent x 3 agents)
        assert by_name["swe-bench"]["n_trials"] == 15
        # Passed totals across agents:
        # swe-bench: 5 + 2 + 3 = 10
        # humaneval: 3 + 4 + 3 = 10
        # mbpp:      2 + 2 + 3 = 7
        # polyglot:  1 + 0 + 3 = 4
        assert by_name["swe-bench"]["passed"] == 10
        assert by_name["swe-bench"]["pass_rate"] == pytest.approx(10 / 15, rel=1e-3)
        assert by_name["humaneval"]["passed"] == 10
        assert by_name["mbpp"]["passed"] == 7
        assert by_name["polyglot"]["passed"] == 4

    def test_sorted_by_n_trials_desc(self) -> None:
        # Build a frame where benchmark counts differ so ordering is unambiguous.
        anns = (
            [_mk(benchmark="big", passed=True, reward=1.0) for _ in range(10)]
            + [_mk(benchmark="med", passed=True, reward=1.0) for _ in range(5)]
            + [_mk(benchmark="small", passed=True, reward=1.0) for _ in range(2)]
        )
        frame = _build_trials_frame(anns)
        result = _per_benchmark_summary(frame)
        assert [r["benchmark"] for r in result] == ["big", "med", "small"]

    def test_top_failures_by_count_within_benchmark(self) -> None:
        # top_failures aggregates categories from FAILED trials only and is
        # capped at 3 entries. Passing trials' categories are excluded.
        def _failed(cat: str) -> dict:
            return _mk(
                benchmark="b1",
                passed=False,
                reward=0.0,
                categories=[{"name": cat, "confidence": 0.9, "evidence": ""}],
            )

        anns = (
            [_failed("cat_a") for _ in range(3)]
            + [_failed("cat_b") for _ in range(5)]
            + [_failed("cat_c")]
            + [
                _mk(
                    benchmark="b1",
                    passed=True,
                    reward=1.0,
                    categories=[
                        {"name": "cat_ignored", "confidence": 0.9, "evidence": ""}
                    ],
                )
            ]
        )
        frame = _build_trials_frame(anns)
        top = _per_benchmark_summary(frame)[0]["top_failures"]
        assert len(top) <= 3
        assert top[0] == {"name": "cat_b", "count": 5}
        assert top[1] == {"name": "cat_a", "count": 3}
        assert top[2] == {"name": "cat_c", "count": 1}
        assert "cat_ignored" not in {t["name"] for t in top}

    def test_mean_reward_aggregation(self, rich_annotations: list[dict]) -> None:
        frame = _build_trials_frame(rich_annotations)
        result = _per_benchmark_summary(frame)
        by_name = {r["benchmark"]: r for r in result}
        # rewards are 1.0 for passes, 0.0 for fails -> mean_reward == pass_rate.
        assert by_name["swe-bench"]["mean_reward"] == pytest.approx(10 / 15, rel=1e-3)
        assert by_name["polyglot"]["mean_reward"] == pytest.approx(4 / 15, rel=1e-3)

    def test_empty_frame_returns_empty_list(self) -> None:
        assert _per_benchmark_summary([]) == []

    def test_missing_benchmark_bucketed_as_unknown(self) -> None:
        anns = [_mk(benchmark=None, passed=True, reward=1.0) for _ in range(3)]
        frame = _build_trials_frame(anns)
        result = _per_benchmark_summary(frame)
        assert result[0]["benchmark"] == "unknown"
        assert result[0]["n_trials"] == 3


# ---------------------------------------------------------------------------
# _per_agent_summary
# ---------------------------------------------------------------------------


class TestPerAgentSummary:
    def test_keyed_by_agent_and_model(self) -> None:
        # Same agent_name, two different models -> two rows.
        anns = [
            _mk(agent="claude-code", model="sonnet-4.6", passed=True, reward=1.0),
            _mk(agent="claude-code", model="opus-4.6", passed=False, reward=0.0),
        ]
        frame = _build_trials_frame(anns)
        result = _per_agent_summary(frame)
        # Two distinct rows, one per (agent, model).
        assert len(result) == 2
        labels = {r["label"] for r in result}
        assert labels == {
            "claude-code / sonnet-4.6",
            "claude-code / opus-4.6",
        }

    def test_single_model_drops_slash(self) -> None:
        anns = [
            _mk(agent="openhands", model="gpt-4", passed=True, reward=1.0)
            for _ in range(3)
        ]
        frame = _build_trials_frame(anns)
        result = _per_agent_summary(frame)
        assert len(result) == 1
        assert result[0]["label"] == "openhands"
        assert result[0]["agent"] == "openhands"
        assert result[0]["model"] == "gpt-4"

    def test_aggregates_pass_rate(self, rich_annotations: list[dict]) -> None:
        frame = _build_trials_frame(rich_annotations)
        result = _per_agent_summary(frame)
        by_label = {r["label"]: r for r in result}
        # claude-code: 5+3+2+1 = 11 passes out of 20
        assert by_label["claude-code"]["passed"] == 11
        assert by_label["claude-code"]["pass_rate"] == pytest.approx(0.55, rel=1e-3)

    def test_missing_agent_bucketed_as_unknown(self) -> None:
        anns = [_mk(agent=None, passed=True, reward=1.0) for _ in range(2)]
        frame = _build_trials_frame(anns)
        result = _per_agent_summary(frame)
        assert result[0]["agent"] == "unknown"
        assert result[0]["n_trials"] == 2

    def test_model_none_does_not_crash(self) -> None:
        anns = [_mk(agent="a1", model=None, passed=True, reward=1.0) for _ in range(3)]
        frame = _build_trials_frame(anns)
        result = _per_agent_summary(frame)
        assert len(result) == 1
        assert result[0]["n_trials"] == 3
        assert result[0]["agent"] == "a1"

    def test_label_is_per_agent_scoped(self) -> None:
        # Agent "a" has two models, agent "b" has one. Only "a" rows get the
        # "/ model" suffix; "b" stays as bare label.
        anns = [
            _mk(agent="a", model="m1", passed=True, reward=1.0),
            _mk(agent="a", model="m2", passed=False, reward=0.0),
            _mk(agent="b", model="m3", passed=True, reward=1.0),
        ]
        frame = _build_trials_frame(anns)
        result = _per_agent_summary(frame)
        by_label = {r["label"]: r for r in result}
        assert "a / m1" in by_label
        assert "a / m2" in by_label
        assert by_label["b"]["agent"] == "b"


# ---------------------------------------------------------------------------
# _agent_benchmark_matrix
# ---------------------------------------------------------------------------


class TestAgentBenchmarkMatrix:
    def test_shape_and_pass_rates(self, rich_annotations: list[dict]) -> None:
        frame = _build_trials_frame(rich_annotations)
        matrix = _agent_benchmark_matrix(frame)
        assert set(matrix.keys()) == {"claude-code", "openhands", "cursor-cli"}
        cc = matrix["claude-code"]
        assert cc["swe-bench"]["pass_rate"] == pytest.approx(1.0)
        assert cc["polyglot"]["pass_rate"] == pytest.approx(0.2)
        assert cc["polyglot"]["n_trials"] == 5
        assert cc["polyglot"]["low_confidence"] is False

    def test_low_confidence_flag(self) -> None:
        # 2 trials in a cell -> low_confidence True.
        anns = [
            _mk(agent="a1", benchmark="b1", passed=True, reward=1.0),
            _mk(agent="a1", benchmark="b1", passed=False, reward=0.0),
        ]
        frame = _build_trials_frame(anns)
        matrix = _agent_benchmark_matrix(frame)
        cell = matrix["a1"]["b1"]
        assert cell["n_trials"] == 2
        assert cell["low_confidence"] is True
        assert cell["pass_rate"] == pytest.approx(0.5)

    def test_exactly_at_threshold_is_not_low_confidence(self) -> None:
        anns = [
            _mk(agent="a1", benchmark="b1", passed=True, reward=1.0)
            for _ in range(LOW_CONFIDENCE_N)
        ]
        frame = _build_trials_frame(anns)
        cell = _agent_benchmark_matrix(frame)["a1"]["b1"]
        assert cell["low_confidence"] is False

    def test_marginalizes_across_configs(self) -> None:
        # Same agent+benchmark across two configs should aggregate into one cell.
        anns = [
            _mk(
                agent="a1",
                benchmark="b1",
                config_name="baseline",
                passed=True,
                reward=1.0,
            ),
            _mk(
                agent="a1",
                benchmark="b1",
                config_name="mcp_tools",
                passed=False,
                reward=0.0,
            ),
            _mk(
                agent="a1",
                benchmark="b1",
                config_name="mcp_tools",
                passed=True,
                reward=1.0,
            ),
        ]
        frame = _build_trials_frame(anns)
        cell = _agent_benchmark_matrix(frame)["a1"]["b1"]
        assert cell["n_trials"] == 3
        assert cell["pass_rate"] == pytest.approx(2 / 3, abs=1e-3)

    def test_returns_plain_dicts(self, rich_annotations: list[dict]) -> None:
        frame = _build_trials_frame(rich_annotations)
        matrix = _agent_benchmark_matrix(frame)
        # Must round-trip through JSON.
        json.dumps(matrix)


# ---------------------------------------------------------------------------
# _top_divergences
# ---------------------------------------------------------------------------


class TestTopDivergences:
    def test_identifies_max_pairwise_delta_per_benchmark(
        self, rich_annotations: list[dict]
    ) -> None:
        frame = _build_trials_frame(rich_annotations)
        matrix = _agent_benchmark_matrix(frame)
        result = _top_divergences(matrix)
        # Spreads per benchmark:
        #   polyglot:  cursor=0.6, claude=0.2, openhands=0.0 -> delta 0.6
        #   swe-bench: claude=1.0, cursor=0.6, openhands=0.4 -> delta 0.6
        #   mbpp:      cursor=0.6, claude=0.4, openhands=0.4 -> delta ~0.2
        #   humaneval: openhands=0.8, claude=0.6, cursor=0.6 -> delta ~0.2
        # Tie between polyglot and swe-bench; alphabetical tiebreaker -> polyglot first.
        top = result[0]
        assert top["benchmark"] == "polyglot"
        assert top["max_delta"] == pytest.approx(0.6)
        assert {top["pair_a"], top["pair_b"]} == {"openhands", "cursor-cli"}
        assert result[1]["benchmark"] == "swe-bench"

    def test_respects_top_k(self, rich_annotations: list[dict]) -> None:
        frame = _build_trials_frame(rich_annotations)
        matrix = _agent_benchmark_matrix(frame)
        assert len(_top_divergences(matrix, top_k=2)) == 2
        assert len(_top_divergences(matrix, top_k=100)) == 4

    def test_deterministic_tiebreaker(self) -> None:
        # Two benchmarks with identical deltas -> alphabetical tiebreaker.
        # Synthesize a small matrix directly.
        matrix = {
            "a1": {
                "beta": {"pass_rate": 1.0, "n_trials": 5, "low_confidence": False},
                "alpha": {"pass_rate": 1.0, "n_trials": 5, "low_confidence": False},
            },
            "a2": {
                "beta": {"pass_rate": 0.0, "n_trials": 5, "low_confidence": False},
                "alpha": {"pass_rate": 0.0, "n_trials": 5, "low_confidence": False},
            },
        }
        result = _top_divergences(matrix)
        # Both have max_delta=1.0; alphabetical -> alpha first.
        assert [r["benchmark"] for r in result] == ["alpha", "beta"]

    def test_includes_full_agent_list(self, rich_annotations: list[dict]) -> None:
        frame = _build_trials_frame(rich_annotations)
        matrix = _agent_benchmark_matrix(frame)
        result = _top_divergences(matrix)
        # Every row should carry the full per-agent list for re-sorting.
        for row in result:
            agents = {a["agent"] for a in row["all_agents"]}
            assert agents == {"claude-code", "openhands", "cursor-cli"}

    def test_skips_benchmarks_with_single_agent(self) -> None:
        matrix = {
            "a1": {"solo": {"pass_rate": 0.5, "n_trials": 5, "low_confidence": False}},
        }
        assert _top_divergences(matrix) == []

    def test_pair_identity_stable_when_top_rates_tie(self) -> None:
        # Two agents share the top pass rate; pair_a / pair_b must be chosen
        # deterministically by agent name ascending.
        matrix = {
            "zeta": {"b": {"pass_rate": 1.0, "n_trials": 5, "low_confidence": False}},
            "alpha": {"b": {"pass_rate": 1.0, "n_trials": 5, "low_confidence": False}},
            "bravo": {"b": {"pass_rate": 0.0, "n_trials": 5, "low_confidence": False}},
        }
        result = _top_divergences(matrix)
        # With ascending name tiebreak on the sort key, alpha wins the top slot.
        assert result[0]["pair_a"] == "alpha"
        assert result[0]["pair_b"] == "bravo"


# ---------------------------------------------------------------------------
# generate_report integration
# ---------------------------------------------------------------------------


class TestGenerateReportIntegration:
    def test_markdown_contains_new_section_headers(
        self, tmp_path: Path, rich_annotations: list[dict]
    ) -> None:
        md_path, _ = generate_report({"annotations": rich_annotations}, tmp_path)
        content = md_path.read_text()
        assert "## Per-Benchmark Summary" in content
        assert "## Per-Agent Summary" in content
        assert "## Agent x Benchmark Matrix" in content
        assert "## Top Divergences" in content

    def test_json_contains_new_keys_with_data(
        self, tmp_path: Path, rich_annotations: list[dict]
    ) -> None:
        _, json_path = generate_report({"annotations": rich_annotations}, tmp_path)
        data = json.loads(json_path.read_text())
        assert len(data["per_benchmark_summary"]) == 4
        assert len(data["per_agent_summary"]) == 3
        assert set(data["agent_benchmark_matrix"].keys()) == {
            "claude-code",
            "openhands",
            "cursor-cli",
        }
        assert len(data["top_divergences"]) == 4

    def test_markdown_escapes_pipe_and_newline_in_names(self, tmp_path: Path) -> None:
        # Adversarial agent/benchmark names must not break table alignment
        # or inject new sections/rows.
        anns = [
            _mk(
                agent="bad|agent\n## Injected",
                benchmark="bench|name",
                passed=True,
                reward=1.0,
            ),
            _mk(
                agent="bad|agent\n## Injected",
                benchmark="bench|name",
                passed=False,
                reward=0.0,
            ),
        ]
        md_path, _ = generate_report({"annotations": anns}, tmp_path)
        content = md_path.read_text()
        # Injected heading must not appear as a real heading.
        assert "\n## Injected" not in content
        # Pipe is escaped as \| so it cannot split the cell.
        assert "bad\\|agent" in content
        assert "bench\\|name" in content

    def test_malformed_category_does_not_abort(self, tmp_path: Path) -> None:
        # A category dict missing the 'name' field must be skipped rather
        # than raise KeyError aborting the whole report.
        anns = [
            _mk(
                passed=False,
                reward=0.0,
                categories=[
                    {"confidence": 0.5, "evidence": "x"},  # no 'name'
                    {"name": "real_cat", "confidence": 0.8, "evidence": "y"},
                ],
            ),
        ]
        md_path, json_path = generate_report({"annotations": anns}, tmp_path)
        data = json.loads(json_path.read_text())
        top = data["per_benchmark_summary"][0]["top_failures"]
        assert {t["name"] for t in top} == {"real_cat"}

    def test_mean_reward_none_when_no_reward_data(self, tmp_path: Path) -> None:
        # When every trial lacks reward data, mean_reward is None (not 0.0)
        # and markdown renders "—".
        anns = [
            {
                "task_id": "t1",
                "agent_name": "a1",
                "benchmark": "b1",
                "passed": True,
                "reward": None,
                "categories": [],
            },
            {
                "task_id": "t2",
                "agent_name": "a1",
                "benchmark": "b1",
                "passed": False,
                "reward": None,
                "categories": [],
            },
        ]
        md_path, json_path = generate_report({"annotations": anns}, tmp_path)
        data = json.loads(json_path.read_text())
        assert data["per_benchmark_summary"][0]["mean_reward"] is None
        assert data["per_agent_summary"][0]["mean_reward"] is None

    def test_low_confidence_footnote_rendered_when_applicable(
        self, tmp_path: Path
    ) -> None:
        # Force one low-N cell.
        anns = [
            _mk(agent="a1", benchmark="b1", passed=True, reward=1.0),
            _mk(agent="a1", benchmark="b1", passed=False, reward=0.0),
        ]
        md_path, _ = generate_report({"annotations": anns}, tmp_path)
        content = md_path.read_text()
        # Footnote marker used in low-confidence rendering.
        assert "†" in content

    def test_new_sections_positioned_near_top(
        self, tmp_path: Path, rich_annotations: list[dict]
    ) -> None:
        md_path, _ = generate_report({"annotations": rich_annotations}, tmp_path)
        content = md_path.read_text()
        # New per-benchmark summary must appear before Trajectory-Dependent
        # (which is the first of the existing detail sections).
        assert content.index("## Per-Benchmark Summary") < content.index(
            "## Trajectory-Dependent Categories"
        )


# ---------------------------------------------------------------------------
# Trajectory-volume metrics (bead agent-diagnostics-fh6)
# ---------------------------------------------------------------------------


class TestTrajectoryVolumeCarry:
    """trajectory_length / total_turns flow from annotations into the trials frame."""

    def test_frame_carries_trajectory_length_and_total_turns(self) -> None:
        anns = [_mk(passed=True, reward=1.0, trajectory_length=42, total_turns=7)]
        frame = _build_trials_frame(anns)
        row = frame[0]
        assert row["trajectory_length"] == 42
        assert row["total_turns"] == 7

    def test_frame_uses_none_when_fields_missing(self) -> None:
        anns = [_mk(passed=True, reward=1.0)]
        frame = _build_trials_frame(anns)
        row = frame[0]
        assert row["trajectory_length"] is None
        assert row["total_turns"] is None

    def test_zero_is_preserved_not_coerced_to_none(self) -> None:
        # 0 is a legitimate trajectory_length (crashed before first turn).
        anns = [_mk(passed=False, reward=0.0, trajectory_length=0, total_turns=0)]
        frame = _build_trials_frame(anns)
        row = frame[0]
        assert row["trajectory_length"] == 0
        assert row["total_turns"] == 0


class TestAggregateTrajectoryMeans:
    """_aggregate_slice exposes mean_trajectory_length / mean_total_turns."""

    def test_mean_trajectory_length_and_total_turns(
        self, rich_annotations: list[dict]
    ) -> None:
        frame = _build_trials_frame(rich_annotations)
        result = _per_benchmark_summary(frame)
        by_name = {r["benchmark"]: r for r in result}
        # Every trial has trajectory_length=10 and total_turns=3.
        assert by_name["swe-bench"]["mean_trajectory_length"] == pytest.approx(10.0)
        assert by_name["swe-bench"]["mean_total_turns"] == pytest.approx(3.0)

    def test_none_excluded_from_mean_denominator(self) -> None:
        # Mix of present and missing values: None must NOT be treated as 0.
        # The third annotation has no trajectory_length AND no total_turns;
        # both means exclude it from both the numerator and denominator.
        anns = [
            _mk(passed=True, reward=1.0, trajectory_length=10, total_turns=2),
            _mk(passed=True, reward=1.0, trajectory_length=20, total_turns=4),
            _mk(passed=False, reward=0.0),  # no trajectory / turns
        ]
        frame = _build_trials_frame(anns)
        result = _per_benchmark_summary(frame)
        row = result[0]
        # Mean trajectory_length of [10, 20] = 15 (None excluded), not (10+20+0)/3.
        # Mean total_turns of [2, 4] = 3 (None excluded), not (2+4+0)/3.
        assert row["mean_trajectory_length"] == pytest.approx(15.0)
        assert row["mean_total_turns"] == pytest.approx(3.0)

    def test_zero_included_in_mean_denominator(self) -> None:
        # 0 is real data — must be averaged, not excluded.
        anns = [
            _mk(passed=False, reward=0.0, trajectory_length=0, total_turns=0),
            _mk(passed=True, reward=1.0, trajectory_length=10, total_turns=4),
        ]
        frame = _build_trials_frame(anns)
        row = _per_benchmark_summary(frame)[0]
        assert row["mean_trajectory_length"] == pytest.approx(5.0)
        assert row["mean_total_turns"] == pytest.approx(2.0)

    def test_mean_is_none_when_no_data(self) -> None:
        anns = [_mk(passed=True, reward=1.0), _mk(passed=False, reward=0.0)]
        frame = _build_trials_frame(anns)
        row = _per_benchmark_summary(frame)[0]
        assert row["mean_trajectory_length"] is None
        assert row["mean_total_turns"] is None

    def test_per_agent_also_aggregates_means(
        self, rich_annotations: list[dict]
    ) -> None:
        frame = _build_trials_frame(rich_annotations)
        result = _per_agent_summary(frame)
        for row in result:
            assert row["mean_trajectory_length"] == pytest.approx(10.0)
            assert row["mean_total_turns"] == pytest.approx(3.0)


class TestTrajectoryRendering:
    """Markdown renders Mean Trajectory Length column; JSON carries both keys."""

    def test_markdown_per_benchmark_has_trajectory_column(
        self, tmp_path: Path, rich_annotations: list[dict]
    ) -> None:
        md_path, _ = generate_report({"annotations": rich_annotations}, tmp_path)
        content = md_path.read_text()
        per_bench_section = content.split("## Per-Benchmark Summary", 1)[1].split(
            "## Per-Agent Summary", 1
        )[0]
        assert "Mean Trajectory Length" in per_bench_section

    def test_markdown_per_agent_has_trajectory_column(
        self, tmp_path: Path, rich_annotations: list[dict]
    ) -> None:
        md_path, _ = generate_report({"annotations": rich_annotations}, tmp_path)
        content = md_path.read_text()
        per_agent_section = content.split("## Per-Agent Summary", 1)[1].split(
            "## Agent x Benchmark Matrix", 1
        )[0]
        assert "Mean Trajectory Length" in per_agent_section

    def test_json_carries_mean_trajectory_length_and_mean_total_turns(
        self, tmp_path: Path, rich_annotations: list[dict]
    ) -> None:
        _, json_path = generate_report({"annotations": rich_annotations}, tmp_path)
        data = json.loads(json_path.read_text())
        for row in data["per_benchmark_summary"]:
            assert "mean_trajectory_length" in row
            assert "mean_total_turns" in row
        for row in data["per_agent_summary"]:
            assert "mean_trajectory_length" in row
            assert "mean_total_turns" in row

    def test_markdown_renders_em_dash_for_none_trajectory(
        self, tmp_path: Path
    ) -> None:
        # Annotations without trajectory_length -> mean is None -> "—".
        anns = [
            _mk(passed=True, reward=1.0),
            _mk(passed=False, reward=0.0),
        ]
        md_path, _ = generate_report({"annotations": anns}, tmp_path)
        content = md_path.read_text()
        per_bench = content.split("## Per-Benchmark Summary", 1)[1].split(
            "## Per-Agent Summary", 1
        )[0]
        # With no trajectory data, the Mean Trajectory Length cell is "—".
        # Count "—" cells — at minimum the trajectory column contributes one per row.
        assert per_bench.count("—") >= 1

    def test_markdown_renders_integer_means_rounded(
        self, tmp_path: Path, rich_annotations: list[dict]
    ) -> None:
        # Means are numeric; rendering should not crash or print 'None'.
        md_path, _ = generate_report({"annotations": rich_annotations}, tmp_path)
        content = md_path.read_text()
        per_bench = content.split("## Per-Benchmark Summary", 1)[1].split(
            "## Per-Agent Summary", 1
        )[0]
        assert "None" not in per_bench
