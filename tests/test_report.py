"""Tests for the agent-observatory report module."""

import json
from pathlib import Path

import pytest

from agent_observatory.report import generate_report


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
        from agent_observatory.report import generate_report as gr

        assert callable(gr)
