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

    def test_category_frequency_section(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        md_path, _ = generate_report(sample_annotations, tmp_path)
        content = md_path.read_text()
        assert "## Category Frequency" in content

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
        assert counts["retrieval_failure"] == 2

    def test_success_via_code_nav_count(
        self, tmp_path: Path, sample_annotations: dict
    ) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        counts = json.loads(json_path.read_text())["category_counts"]
        assert counts["success_via_code_nav"] == 2

    def test_query_churn_count(self, tmp_path: Path, sample_annotations: dict) -> None:
        _, json_path = generate_report(sample_annotations, tmp_path)
        counts = json.loads(json_path.read_text())["category_counts"]
        assert counts["query_churn"] == 1


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

    def test_empty_md_has_sections(self, tmp_path: Path) -> None:
        md_path, _ = generate_report({"annotations": []}, tmp_path)
        content = md_path.read_text()
        assert "## Corpus Statistics" in content
        assert "## Category Frequency" in content
        assert "## Success Mode Summary" in content


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
