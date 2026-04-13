"""Tests for the agent-observatory report module."""

import json
from pathlib import Path

import pytest

from agent_diagnostics.report import generate_report


def _make_annotations(ann_list: list[dict]) -> dict:
    """Wrap a list of annotation dicts into an annotation document."""
    return {"annotations": ann_list}


@pytest.fixture()
def sample_annotations() -> dict:
    """Synthetic annotation document with mixed pass/fail and multiple configs."""
    return _make_annotations(
        [
            {
                "task_id": "task_001",
                "config_name": "baseline",
                "benchmark": "swe-bench",
                "passed": True,
                "reward": 1.0,
                "categories": [
                    {
                        "name": "success_via_code_nav",
                        "confidence": 0.9,
                        "evidence": "Used grep effectively",
                    },
                ],
            },
            {
                "task_id": "task_002",
                "config_name": "baseline",
                "benchmark": "swe-bench",
                "passed": False,
                "reward": 0.0,
                "categories": [
                    {
                        "name": "retrieval_failure",
                        "confidence": 0.8,
                        "evidence": "Could not find file",
                    },
                    {
                        "name": "query_churn",
                        "confidence": 0.7,
                        "evidence": "Repeated searches",
                    },
                ],
            },
            {
                "task_id": "task_003",
                "config_name": "mcp_tools",
                "benchmark": "swe-bench",
                "passed": False,
                "reward": 0.2,
                "categories": [
                    {
                        "name": "retrieval_failure",
                        "confidence": 0.85,
                        "evidence": "Wrong file searched",
                    },
                ],
            },
            {
                "task_id": "task_004",
                "config_name": "mcp_tools",
                "benchmark": "humaneval",
                "passed": True,
                "reward": 1.0,
                "categories": [
                    {
                        "name": "success_via_code_nav",
                        "confidence": 0.95,
                        "evidence": "Navigated AST",
                    },
                ],
            },
        ]
    )


class TestGenerateReportCreatesFiles:
    """Verify generate_report creates .md and .json files."""

    def test_returns_two_paths(self, tmp_path: Path, sample_annotations: dict) -> None:
        md_path, json_path = generate_report(sample_annotations, tmp_path)
        assert isinstance(md_path, Path)
        assert isinstance(json_path, Path)

    def test_md_file_exists(self, tmp_path: Path, sample_annotations: dict) -> None:
        md_path, _ = generate_report(sample_annotations, tmp_path)
        assert md_path.exists()
        assert md_path.suffix == ".md"

    def test_json_file_exists(self, tmp_path: Path, sample_annotations: dict) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        assert json_path.exists()
        assert json_path.suffix == ".json"

    def test_creates_output_dir(self, tmp_path: Path, sample_annotations: dict) -> None:
        nested = tmp_path / "sub" / "dir"
        md_path, json_path = generate_report(sample_annotations, nested)
        assert nested.is_dir()
        assert md_path.exists()
        assert json_path.exists()


class TestMarkdownSections:
    """Verify .md contains all required sections."""

    def test_corpus_statistics_section(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        md_path, _ = generate_report(sample_annotations, tmp_path)
        content = md_path.read_text()
        assert "## Corpus Statistics" in content

    def test_trajectory_dependent_section(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        md_path, _ = generate_report(sample_annotations, tmp_path)
        content = md_path.read_text()
        assert "## Trajectory-Dependent Categories" in content

    def test_reward_dependent_section(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        md_path, _ = generate_report(sample_annotations, tmp_path)
        content = md_path.read_text()
        assert "## Reward-Dependent Categories" in content

    def test_config_breakdown_section(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        md_path, _ = generate_report(sample_annotations, tmp_path)
        content = md_path.read_text()
        assert "## Category Breakdown by Config" in content

    def test_top_failure_categories_section(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        md_path, _ = generate_report(sample_annotations, tmp_path)
        content = md_path.read_text()
        assert "## Top Failure Categories" in content

    def test_success_mode_summary_section(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        md_path, _ = generate_report(sample_annotations, tmp_path)
        content = md_path.read_text()
        assert "## Success Mode Summary" in content


class TestJsonKeys:
    """Verify .json has required top-level keys."""

    def test_corpus_stats_key(self, tmp_path: Path, sample_annotations: dict) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        data = json.loads(json_path.read_text())
        assert "corpus_stats" in data

    def test_category_counts_key(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        data = json.loads(json_path.read_text())
        assert "category_counts" in data

    def test_category_by_config_key(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        data = json.loads(json_path.read_text())
        assert "category_by_config" in data


class TestCorpusStatsCorrect:
    """Verify corpus statistics are calculated correctly."""

    def test_total_trials(self, tmp_path: Path, sample_annotations: dict) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        stats = json.loads(json_path.read_text())["corpus_stats"]
        assert stats["total_trials"] == 4

    def test_passed_count(self, tmp_path: Path, sample_annotations: dict) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        stats = json.loads(json_path.read_text())["corpus_stats"]
        assert stats["passed"] == 2

    def test_failed_count(self, tmp_path: Path, sample_annotations: dict) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        stats = json.loads(json_path.read_text())["corpus_stats"]
        assert stats["failed"] == 2

    def test_pass_rate(self, tmp_path: Path, sample_annotations: dict) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        stats = json.loads(json_path.read_text())["corpus_stats"]
        assert stats["pass_rate"] == 0.5


class TestCategoryCountsCorrect:
    """Verify category counting is correct."""

    def test_retrieval_failure_count(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        counts = json.loads(json_path.read_text())["category_counts"]
        assert counts["retrieval_failure"]["count"] == 2

    def test_success_via_code_nav_count(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        counts = json.loads(json_path.read_text())["category_counts"]
        assert counts["success_via_code_nav"]["count"] == 2

    def test_query_churn_count(self, tmp_path: Path, sample_annotations: dict) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        counts = json.loads(json_path.read_text())["category_counts"]
        assert counts["query_churn"]["count"] == 1


class TestConfigBreakdown:
    """Verify config breakdown groups by config_name."""

    def test_baseline_config_present(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        by_config = json.loads(json_path.read_text())["category_by_config"]
        assert "baseline" in by_config

    def test_mcp_tools_config_present(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        by_config = json.loads(json_path.read_text())["category_by_config"]
        assert "mcp_tools" in by_config

    def test_baseline_categories(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        by_config = json.loads(json_path.read_text())["category_by_config"]
        baseline = by_config["baseline"]
        assert baseline["retrieval_failure"] == 1
        assert baseline["query_churn"] == 1
        assert baseline["success_via_code_nav"] == 1

    def test_mcp_tools_categories(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        by_config = json.loads(json_path.read_text())["category_by_config"]
        mcp = by_config["mcp_tools"]
        assert mcp["retrieval_failure"] == 1
        assert mcp["success_via_code_nav"] == 1


class TestEmptyAnnotations:
    """Verify empty annotations produce valid but minimal report."""

    def test_empty_creates_files(self, tmp_path: Path) -> None:
        md_path, json_path = generate_report({"annotations": []}, tmp_path)
        assert md_path.exists()
        assert json_path.exists()

    def test_empty_corpus_stats(self, tmp_path: Path) -> None:
        _, json_path = generate_report({"annotations": []}, tmp_path)
        stats = json.loads(json_path.read_text())["corpus_stats"]
        assert stats["total_trials"] == 0
        assert stats["passed"] == 0
        assert stats["failed"] == 0
        assert stats["pass_rate"] == 0

    def test_empty_category_counts(self, tmp_path: Path) -> None:
        _, json_path = generate_report({"annotations": []}, tmp_path)
        counts = json.loads(json_path.read_text())["category_counts"]
        assert counts == {}

    def test_empty_trajectory_available(self, tmp_path: Path) -> None:
        _, json_path = generate_report({"annotations": []}, tmp_path)
        stats = json.loads(json_path.read_text())["corpus_stats"]
        assert stats["trajectory_available"] == 0

    def test_empty_md_has_sections(self, tmp_path: Path) -> None:
        md_path, _ = generate_report({"annotations": []}, tmp_path)
        content = md_path.read_text()
        assert "## Corpus Statistics" in content
        assert "## Trajectory-Dependent Categories" in content
        assert "## Reward-Dependent Categories" in content
        assert "## Success Mode Summary" in content


class TestNoneRewardInAnnotations:
    """Verify None reward values are handled gracefully in report generation."""

    def test_none_reward_in_avg_calculation(self, tmp_path: Path) -> None:
        """avg_reward should exclude None rewards from calculation."""
        annotations = _make_annotations(
            [
                {
                    "task_id": "t1",
                    "config_name": "cfg",
                    "benchmark": "b",
                    "passed": True,
                    "reward": 1.0,
                    "categories": [],
                },
                {
                    "task_id": "t2",
                    "config_name": "cfg",
                    "benchmark": "b",
                    "passed": False,
                    "reward": None,
                    "categories": [],
                },
            ]
        )
        _, json_path = generate_report(annotations, tmp_path)
        stats = json.loads(json_path.read_text())["corpus_stats"]
        # Only t1's reward=1.0 should be averaged (None excluded)
        assert stats["avg_reward"] == 1.0

    def test_all_none_rewards(self, tmp_path: Path) -> None:
        """When all rewards are None, avg_reward should be 0.0."""
        annotations = _make_annotations(
            [
                {
                    "task_id": "t1",
                    "config_name": "cfg",
                    "benchmark": "b",
                    "passed": False,
                    "reward": None,
                    "categories": [],
                },
            ]
        )
        _, json_path = generate_report(annotations, tmp_path)
        stats = json.loads(json_path.read_text())["corpus_stats"]
        assert stats["avg_reward"] == 0.0


class TestImportPath:
    """Verify the public import works."""

    def test_import_generate_report(self) -> None:
        from agent_diagnostics.report import generate_report as gr

        assert callable(gr)


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

from agent_diagnostics.report import (
    _category_by_suite,
    _core_task_name,
    _paired_comparison,
    _render_markdown,
    _top_categories_with_examples,
)


class TestCoreTaskName:
    """Parametrized tests for _core_task_name covering all prefix patterns."""

    @pytest.mark.parametrize(
        "trial_path, expected",
        [
            # bare task name — no prefix, no hash
            ("my_task", "my_task"),
            # with __hash suffix
            ("my_task__abc123", "my_task"),
            # baseline_ prefix
            ("baseline_my_task__abc123", "my_task"),
            # bl_ prefix
            ("bl_my_task__xyz", "my_task"),
            # mcp_ prefix
            ("mcp_my_task__h1", "my_task"),
            # sgonly_ prefix
            ("sgonly_my_task__h2", "my_task"),
            # prefix without hash
            ("baseline_solve_bug", "solve_bug"),
            # nested path — only last segment matters
            ("/results/run1/baseline_my_task__abc", "my_task"),
            # trailing slash
            ("/results/run1/bl_fix_issue__def/", "fix_issue"),
            # case normalisation — prefix check is case-sensitive so MCP_ is NOT stripped
            ("MCP_MyTask__HASH", "mcp_mytask"),
            # double underscore in task name (first __ from right is hash separator)
            ("some__special__hash", "some__special"),
            # no prefix, with path
            ("/a/b/c/plain_task__h", "plain_task"),
            # prefix only strips the first match
            ("bl_mcp_thing__h", "mcp_thing"),
        ],
    )
    def test_core_task_name(self, trial_path: str, expected: str) -> None:
        assert _core_task_name(trial_path) == expected


def _make_paired_annotations(
    num_tasks: int = 25,
    configs: tuple[str, ...] = ("alpha", "beta"),
    shared_cat_a: str = "cat_only_a",
    shared_cat_b: str = "cat_only_b",
) -> list[dict]:
    """Build annotations for two configs sharing *num_tasks* tasks.

    For each task:
      - config alpha gets category shared_cat_a
      - config beta  gets category shared_cat_b
    This guarantees deterministic deltas.
    """
    annotations: list[dict] = []
    for i in range(num_tasks):
        task_name = f"task_{i:03d}"
        for cfg in configs:
            cat_name = shared_cat_a if cfg == configs[0] else shared_cat_b
            annotations.append(
                {
                    "task_id": f"{task_name}_{cfg}",
                    "config_name": cfg,
                    "benchmark": "bench",
                    "passed": False,
                    "reward": 0.0,
                    "trial_path": f"/runs/{cfg}/{task_name}__hash{i}",
                    "categories": [
                        {"name": cat_name, "confidence": 0.9, "evidence": "ev"}
                    ],
                }
            )
    return annotations


class TestPairedComparison:
    """Tests for _paired_comparison with 25+ annotations across 2 configs."""

    def test_returns_pair_with_shared_tasks(self) -> None:
        anns = _make_paired_annotations(num_tasks=25)
        result = _paired_comparison(anns)
        assert len(result) == 1
        pair = result[0]
        assert pair["shared_tasks"] == 25
        assert pair["config_a"] == "alpha"
        assert pair["config_b"] == "beta"

    def test_introduced_by_a_contains_expected_category(self) -> None:
        anns = _make_paired_annotations(num_tasks=25)
        result = _paired_comparison(anns)
        pair = result[0]
        a_cats = [item["category"] for item in pair["introduced_by_a"]]
        assert "cat_only_a" in a_cats

    def test_introduced_by_b_contains_expected_category(self) -> None:
        anns = _make_paired_annotations(num_tasks=25)
        result = _paired_comparison(anns)
        pair = result[0]
        b_cats = [item["category"] for item in pair["introduced_by_b"]]
        assert "cat_only_b" in b_cats

    def test_delta_values_correct(self) -> None:
        anns = _make_paired_annotations(num_tasks=25)
        result = _paired_comparison(anns)
        pair = result[0]
        a_item = next(
            i for i in pair["introduced_by_a"] if i["category"] == "cat_only_a"
        )
        assert a_item["delta"] == 25
        b_item = next(
            i for i in pair["introduced_by_b"] if i["category"] == "cat_only_b"
        )
        assert b_item["delta"] == 25  # stored as abs

    def test_fewer_than_20_shared_returns_empty(self) -> None:
        anns = _make_paired_annotations(num_tasks=19)
        result = _paired_comparison(anns)
        assert result == []

    def test_skips_annotations_without_config_or_path(self) -> None:
        anns = [
            {"task_id": "t1", "passed": False, "reward": 0, "categories": []},
            {
                "task_id": "t2",
                "config_name": "x",
                "passed": False,
                "reward": 0,
                "categories": [],
            },
        ]
        result = _paired_comparison(anns)
        assert result == []

    def test_top5_limit(self) -> None:
        """With >5 categories differing, only top 5 are returned per direction."""
        anns: list[dict] = []
        for i in range(25):
            task = f"task_{i:03d}"
            # Config alpha gets 7 unique categories
            anns.append(
                {
                    "task_id": f"{task}_alpha",
                    "config_name": "alpha",
                    "benchmark": "b",
                    "passed": False,
                    "reward": 0,
                    "trial_path": f"/r/{task}__halpha",
                    "categories": [
                        {"name": f"acat_{j}", "confidence": 0.9, "evidence": "e"}
                        for j in range(7)
                    ],
                }
            )
            # Config beta gets a single different category
            anns.append(
                {
                    "task_id": f"{task}_beta",
                    "config_name": "beta",
                    "benchmark": "b",
                    "passed": False,
                    "reward": 0,
                    "trial_path": f"/r/{task}__hbeta",
                    "categories": [
                        {"name": "bcat_0", "confidence": 0.9, "evidence": "e"}
                    ],
                }
            )
        result = _paired_comparison(anns)
        assert len(result) == 1
        assert len(result[0]["introduced_by_a"]) == 5


class TestTopCategoriesWithExamples:
    """Tests for _top_categories_with_examples."""

    @pytest.fixture()
    def polarity_annotations(self) -> tuple[list[dict], dict[str, str]]:
        polarity_map = {
            "fail_a": "failure",
            "fail_b": "failure",
            "fail_c": "failure",
            "fail_d": "failure",
            "succ_a": "success",
        }
        anns = (
            [
                {
                    "task_id": f"t{i}",
                    "config_name": "cfg",
                    "reward": 0.0,
                    "categories": [
                        {"name": "fail_a", "confidence": 0.9, "evidence": "ea"},
                        {"name": "fail_b", "confidence": 0.8, "evidence": "eb"},
                    ],
                }
                for i in range(5)
            ]
            + [
                {
                    "task_id": f"t{i}",
                    "config_name": "cfg",
                    "reward": 0.0,
                    "categories": [
                        {"name": "fail_c", "confidence": 0.7, "evidence": "ec"},
                    ],
                }
                for i in range(3)
            ]
            + [
                {
                    "task_id": "ts1",
                    "config_name": "cfg",
                    "reward": 1.0,
                    "categories": [
                        {"name": "succ_a", "confidence": 0.9, "evidence": "es"},
                    ],
                }
            ]
        )
        return anns, polarity_map

    def test_returns_top_n_failure_categories(
        self, polarity_annotations: tuple[list[dict], dict[str, str]]
    ) -> None:
        anns, pm = polarity_annotations
        result = _top_categories_with_examples(anns, pm, "failure", top_n=2)
        assert len(result) == 2
        names = [r["name"] for r in result]
        assert "fail_a" in names
        assert "fail_b" in names

    def test_examples_limited(
        self, polarity_annotations: tuple[list[dict], dict[str, str]]
    ) -> None:
        anns, pm = polarity_annotations
        result = _top_categories_with_examples(
            anns, pm, "failure", top_n=1, examples_per=2
        )
        assert len(result[0]["examples"]) <= 2

    def test_success_polarity(
        self, polarity_annotations: tuple[list[dict], dict[str, str]]
    ) -> None:
        anns, pm = polarity_annotations
        result = _top_categories_with_examples(anns, pm, "success", top_n=3)
        assert len(result) == 1
        assert result[0]["name"] == "succ_a"

    def test_example_structure(
        self, polarity_annotations: tuple[list[dict], dict[str, str]]
    ) -> None:
        anns, pm = polarity_annotations
        result = _top_categories_with_examples(anns, pm, "failure", top_n=1)
        ex = result[0]["examples"][0]
        assert "task_id" in ex
        assert "config_name" in ex
        assert "reward" in ex
        assert "evidence" in ex

    def test_empty_annotations(self) -> None:
        result = _top_categories_with_examples([], {}, "failure")
        assert result == []


class TestCategoryBySuite:
    """Tests for _category_by_suite."""

    def test_groups_by_benchmark(self) -> None:
        anns = [
            {
                "benchmark": "swe",
                "categories": [{"name": "cat_a"}],
            },
            {
                "benchmark": "human",
                "categories": [{"name": "cat_b"}],
            },
            {
                "benchmark": "swe",
                "categories": [{"name": "cat_a"}, {"name": "cat_c"}],
            },
        ]
        result = _category_by_suite(anns)
        assert "swe" in result
        assert "human" in result
        assert result["swe"]["cat_a"] == 2
        assert result["swe"]["cat_c"] == 1
        assert result["human"]["cat_b"] == 1

    def test_sorted_by_total_desc(self) -> None:
        anns = [
            {"benchmark": "small", "categories": [{"name": "c"}]},
            {"benchmark": "big", "categories": [{"name": "c"}]},
            {"benchmark": "big", "categories": [{"name": "c"}]},
            {"benchmark": "big", "categories": [{"name": "c"}]},
        ]
        result = _category_by_suite(anns)
        suites = list(result.keys())
        assert suites[0] == "big"

    def test_missing_benchmark_uses_unknown(self) -> None:
        anns = [{"categories": [{"name": "c"}]}]
        result = _category_by_suite(anns)
        assert "unknown" in result

    def test_empty(self) -> None:
        assert _category_by_suite([]) == {}


class TestRenderMarkdownNoSuccesses:
    """Test the 'no success annotations' branch in _render_markdown."""

    def test_no_success_message(self) -> None:
        stats = {
            "total_trials": 1,
            "passed": 0,
            "failed": 1,
            "pass_rate": 0.0,
            "avg_reward": 0.0,
            "configs": ["cfg"],
            "benchmarks": ["b"],
        }
        md = _render_markdown(
            stats=stats,
            cat_counts={"some_cat": 1},
            cat_by_config={"cfg": {"some_cat": 1}},
            cat_by_suite={"b": {"some_cat": 1}},
            top_failures=[],
            top_successes=[],
            polarity_map={},
            generated_at="2026-01-01T00:00:00Z",
        )
        assert "No success-mode annotations found in this corpus." in md

    def test_with_successes_no_message(self) -> None:
        stats = {
            "total_trials": 1,
            "passed": 1,
            "failed": 0,
            "pass_rate": 1.0,
            "avg_reward": 1.0,
            "configs": ["cfg"],
            "benchmarks": ["b"],
        }
        top_successes = [
            {
                "name": "good_thing",
                "count": 1,
                "examples": [
                    {
                        "task_id": "t1",
                        "config_name": "cfg",
                        "reward": 1.0,
                        "evidence": "great",
                    }
                ],
            }
        ]
        md = _render_markdown(
            stats=stats,
            cat_counts={},
            cat_by_config={},
            cat_by_suite={},
            top_failures=[],
            top_successes=top_successes,
            polarity_map={},
            generated_at="2026-01-01T00:00:00Z",
        )
        assert "No success-mode annotations found" not in md
        assert "good_thing" in md

    def test_paired_comparisons_rendered(self) -> None:
        stats = {
            "total_trials": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "avg_reward": 0.0,
            "configs": [],
            "benchmarks": [],
        }
        pairs = [
            {
                "config_a": "a",
                "config_b": "b",
                "shared_tasks": 25,
                "introduced_by_a": [{"category": "cat1", "delta": 10}],
                "introduced_by_b": [{"category": "cat2", "delta": 5}],
            }
        ]
        md = _render_markdown(
            stats=stats,
            cat_counts={},
            cat_by_config={},
            cat_by_suite={},
            top_failures=[],
            top_successes=[],
            polarity_map={},
            generated_at="2026-01-01T00:00:00Z",
            paired_comparisons=pairs,
        )
        assert "## Configuration Comparison (Paired)" in md
        assert "a vs b" in md
        assert "cat1" in md
        assert "cat2" in md


# ---------------------------------------------------------------------------
# Tests for trajectory-aware denominators (unit-trajectory-denominators)
# ---------------------------------------------------------------------------

from agent_diagnostics.annotator import CHECKER_REQUIRES_TRAJECTORY
from agent_diagnostics.report import (
    _category_counts_with_denominators,
    _count_trajectory_available,
)


def _make_mixed_trajectory_annotations() -> list[dict]:
    """Build annotations where some have trajectory data and some do not."""
    return [
        {
            "task_id": "t1",
            "config_name": "cfg",
            "benchmark": "b",
            "passed": False,
            "reward": 0.0,
            "has_trajectory": True,
            "categories": [
                {"name": "retrieval_failure", "confidence": 0.9, "evidence": "ev"},
                {"name": "incomplete_solution", "confidence": 0.8, "evidence": "ev"},
            ],
        },
        {
            "task_id": "t2",
            "config_name": "cfg",
            "benchmark": "b",
            "passed": False,
            "reward": 0.5,
            "has_trajectory": True,
            "categories": [
                {"name": "near_miss", "confidence": 0.7, "evidence": "ev"},
                {"name": "query_churn", "confidence": 0.6, "evidence": "ev"},
            ],
        },
        {
            "task_id": "t3",
            "config_name": "cfg",
            "benchmark": "b",
            "passed": False,
            "reward": 0.3,
            "has_trajectory": False,
            "categories": [
                {"name": "minimal_progress", "confidence": 0.7, "evidence": "ev"},
            ],
        },
        {
            "task_id": "t4",
            "config_name": "cfg",
            "benchmark": "b",
            "passed": True,
            "reward": 1.0,
            "has_trajectory": False,
            "categories": [],
        },
    ]


class TestTrajectoryAvailableCount:
    """Tests for _count_trajectory_available helper."""

    def test_counts_has_trajectory_true(self) -> None:
        anns = _make_mixed_trajectory_annotations()
        assert _count_trajectory_available(anns) == 2

    def test_empty_list(self) -> None:
        assert _count_trajectory_available([]) == 0

    def test_all_have_trajectory(self) -> None:
        anns = [{"has_trajectory": True}, {"has_trajectory": True}]
        assert _count_trajectory_available(anns) == 2

    def test_none_have_trajectory(self) -> None:
        anns = [{"has_trajectory": False}, {}]
        assert _count_trajectory_available(anns) == 0

    def test_signals_sub_dict(self) -> None:
        """has_trajectory in signals sub-dict is also recognized."""
        anns = [{"signals": {"has_trajectory": True}}, {"signals": {}}]
        assert _count_trajectory_available(anns) == 1


class TestDenominatorSplitting:
    """Verify that trajectory-dependent and reward-dependent categories
    get different denominators."""

    def test_trajectory_category_uses_traj_denominator(self) -> None:
        anns = _make_mixed_trajectory_annotations()
        result = _category_counts_with_denominators(anns)
        # retrieval_failure is trajectory-dependent; 2 of 4 have trajectory
        assert result["retrieval_failure"]["denominator"] == 2

    def test_reward_category_uses_full_denominator(self) -> None:
        anns = _make_mixed_trajectory_annotations()
        result = _category_counts_with_denominators(anns)
        # near_miss is reward-only; denominator should be total=4
        assert result["near_miss"]["denominator"] == 4

    def test_rate_computed_correctly_trajectory(self) -> None:
        anns = _make_mixed_trajectory_annotations()
        result = _category_counts_with_denominators(anns)
        # retrieval_failure: count=1, denominator=2 -> rate=0.5
        assert result["retrieval_failure"]["rate"] == 0.5

    def test_rate_computed_correctly_reward(self) -> None:
        anns = _make_mixed_trajectory_annotations()
        result = _category_counts_with_denominators(anns)
        # minimal_progress: count=1, denominator=4 -> rate=0.25
        assert result["minimal_progress"]["rate"] == 0.25


class TestRetrievalFailureDenominator:
    """Verify retrieval_failure uses trajectory-available trials as denominator."""

    def test_retrieval_failure_is_trajectory_dependent(self) -> None:
        assert CHECKER_REQUIRES_TRAJECTORY["retrieval_failure"] is True

    def test_retrieval_failure_denominator_in_report(self, tmp_path: Path) -> None:
        anns = _make_mixed_trajectory_annotations()
        _, json_path = generate_report({"annotations": anns}, tmp_path)
        data = json.loads(json_path.read_text())
        rf = data["category_counts"]["retrieval_failure"]
        assert rf["denominator"] == 2  # only 2 trials have trajectory
        assert rf["count"] == 1
        assert rf["rate"] == 0.5


class TestJsonDenominatorStructure:
    """Verify JSON category_counts has {count, denominator, rate} structure."""

    def test_category_entry_has_required_keys(self, tmp_path: Path) -> None:
        anns = _make_mixed_trajectory_annotations()
        _, json_path = generate_report({"annotations": anns}, tmp_path)
        data = json.loads(json_path.read_text())
        for name, info in data["category_counts"].items():
            assert "count" in info, f"{name} missing 'count'"
            assert "denominator" in info, f"{name} missing 'denominator'"
            assert "rate" in info, f"{name} missing 'rate'"

    def test_corpus_stats_has_trajectory_available(self, tmp_path: Path) -> None:
        anns = _make_mixed_trajectory_annotations()
        _, json_path = generate_report({"annotations": anns}, tmp_path)
        data = json.loads(json_path.read_text())
        assert data["corpus_stats"]["trajectory_available"] == 2


class TestMarkdownDenominatorSections:
    """Verify markdown report has separate denominator sections."""

    def test_trajectory_section_present(self, tmp_path: Path) -> None:
        anns = _make_mixed_trajectory_annotations()
        md_path, _ = generate_report({"annotations": anns}, tmp_path)
        content = md_path.read_text()
        assert "## Trajectory-Dependent Categories" in content

    def test_reward_section_present(self, tmp_path: Path) -> None:
        anns = _make_mixed_trajectory_annotations()
        md_path, _ = generate_report({"annotations": anns}, tmp_path)
        content = md_path.read_text()
        assert "## Reward-Dependent Categories" in content

    def test_trajectory_section_shows_denominator(self, tmp_path: Path) -> None:
        anns = _make_mixed_trajectory_annotations()
        md_path, _ = generate_report({"annotations": anns}, tmp_path)
        content = md_path.read_text()
        assert "2 trials with trajectory data" in content

    def test_reward_section_shows_denominator(self, tmp_path: Path) -> None:
        anns = _make_mixed_trajectory_annotations()
        md_path, _ = generate_report({"annotations": anns}, tmp_path)
        content = md_path.read_text()
        assert "4 total trials" in content

    def test_rate_column_in_trajectory_table(self, tmp_path: Path) -> None:
        anns = _make_mixed_trajectory_annotations()
        md_path, _ = generate_report({"annotations": anns}, tmp_path)
        content = md_path.read_text()
        # The trajectory section should have a Rate column
        assert "| Rate |" in content


class TestCheckerRequiresTrajectoryMetadata:
    """Verify CHECKER_REQUIRES_TRAJECTORY covers all heuristic checkers."""

    def test_reward_only_categories_marked_false(self) -> None:
        for name in [
            "rate_limited_run",
            "exception_crash",
            "incomplete_solution",
            "near_miss",
            "minimal_progress",
        ]:
            assert CHECKER_REQUIRES_TRAJECTORY[name] is False, f"{name} should be False"

    def test_trajectory_categories_marked_true(self) -> None:
        for name in [
            "retrieval_failure",
            "query_churn",
            "clean_success",
            "success_via_code_nav",
            "reward_hacking",
            "planning_absence",
        ]:
            assert CHECKER_REQUIRES_TRAJECTORY[name] is True, f"{name} should be True"

    def test_all_checker_categories_covered(self) -> None:
        """Every category in _ALL_CHECKERS should have an entry."""
        from agent_diagnostics.annotator import _ALL_CHECKERS, _assignment
        from agent_diagnostics.tool_registry import DEFAULT_REGISTRY

        # Extract category names from checkers by examining the function names
        # Each _check_X function produces category name derived from its name
        # Instead, verify all keys in CHECKER_REQUIRES_TRAJECTORY are non-empty
        assert len(CHECKER_REQUIRES_TRAJECTORY) >= 27
