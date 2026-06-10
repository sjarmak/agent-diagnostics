"""Tests for agent_diagnostics.agreement (inter-annotator Cohen's kappa)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytest

from agent_diagnostics.agreement import (
    SCHEMA_VERSION,
    cohens_kappa,
    compute_agreement,
    format_markdown,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(trial_id: str, category: str, identity: str, annotator_type: str = "heuristic") -> dict:
    return {
        "trial_id": trial_id,
        "category_name": category,
        "confidence": 0.9,
        "evidence": "test",
        "annotator_type": annotator_type,
        "annotator_identity": identity,
        "taxonomy_version": "3.0",
        "annotated_at": "2026-06-09T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Tests — cohens_kappa
# ---------------------------------------------------------------------------


class TestCohensKappa:
    def test_perfect_agreement(self) -> None:
        assert cohens_kappa(n11=5, n10=0, n01=0, n00=5) == pytest.approx(1.0)

    def test_complete_disagreement(self) -> None:
        assert cohens_kappa(n11=0, n10=5, n01=5, n00=0) == pytest.approx(-1.0)

    def test_chance_level_is_zero(self) -> None:
        # Independent annotators, each saying yes half the time.
        assert cohens_kappa(n11=25, n10=25, n01=25, n00=25) == pytest.approx(0.0)

    def test_undefined_when_both_constant(self) -> None:
        # Both always assign → chance agreement is 1.0.
        assert cohens_kappa(n11=10, n10=0, n01=0, n00=0) is None
        # Both never assign.
        assert cohens_kappa(n11=0, n10=0, n01=0, n00=10) is None

    def test_empty_table_is_undefined(self) -> None:
        assert cohens_kappa(0, 0, 0, 0) is None


# ---------------------------------------------------------------------------
# Tests — compute_agreement
# ---------------------------------------------------------------------------


class TestComputeAgreement:
    def test_perfect_agreement_pair(self) -> None:
        rows = [
            _row("t1", "retrieval_failure", "heuristic:rule-engine"),
            _row("t2", "query_churn", "heuristic:rule-engine"),
            _row("t1", "retrieval_failure", "llm:haiku-4", "llm"),
            _row("t2", "query_churn", "llm:haiku-4", "llm"),
        ]
        summary = compute_agreement(rows)

        assert summary["schema_version"] == SCHEMA_VERSION
        assert summary["annotators"] == {"heuristic:rule-engine": 2, "llm:haiku-4": 2}
        assert len(summary["pairs"]) == 1

        pair = summary["pairs"][0]
        assert pair["shared_trials"] == 2
        assert pair["categories"]["retrieval_failure"]["kappa"] == pytest.approx(1.0)
        assert pair["categories"]["query_churn"]["kappa"] == pytest.approx(1.0)
        assert pair["mean_kappa"] == pytest.approx(1.0)

    def test_systematic_disagreement_is_negative(self) -> None:
        rows = [
            _row("t1", "retrieval_failure", "a:1"),
            _row("t2", "other_cat", "a:1"),
            _row("t2", "retrieval_failure", "b:2"),
            _row("t1", "other_cat", "b:2"),
        ]
        summary = compute_agreement(rows)
        pair = summary["pairs"][0]
        assert pair["categories"]["retrieval_failure"]["kappa"] == pytest.approx(-1.0)

    def test_constant_category_is_undefined(self) -> None:
        # Both annotators assign the category on every shared trial.
        rows = [
            _row("t1", "exception_crash", "a:1"),
            _row("t2", "exception_crash", "a:1"),
            _row("t1", "exception_crash", "b:2"),
            _row("t2", "exception_crash", "b:2"),
        ]
        summary = compute_agreement(rows)
        pair = summary["pairs"][0]
        assert pair["categories"]["exception_crash"]["kappa"] is None
        assert pair["mean_kappa"] is None

    def test_single_identity_yields_no_pairs(self) -> None:
        rows = [_row("t1", "retrieval_failure", "a:1")]
        summary = compute_agreement(rows)
        assert summary["pairs"] == []

    def test_disjoint_trials_yield_empty_categories(self) -> None:
        rows = [
            _row("t1", "retrieval_failure", "a:1"),
            _row("t2", "retrieval_failure", "b:2"),
        ]
        summary = compute_agreement(rows)
        pair = summary["pairs"][0]
        assert pair["shared_trials"] == 0
        assert pair["categories"] == {}
        assert pair["mean_kappa"] is None

    def test_contingency_counts(self) -> None:
        rows = [
            # t1: both assign; t2: only a; t3: only b; t4: neither (but both
            # annotated t4 via another category, keeping it in the universe).
            _row("t1", "cat_x", "a:1"),
            _row("t2", "cat_x", "a:1"),
            _row("t3", "pad", "a:1"),
            _row("t4", "pad", "a:1"),
            _row("t1", "cat_x", "b:2"),
            _row("t3", "cat_x", "b:2"),
            _row("t2", "pad", "b:2"),
            _row("t4", "pad", "b:2"),
        ]
        summary = compute_agreement(rows)
        c = summary["pairs"][0]["categories"]["cat_x"]
        assert (c["both_present"], c["only_a"], c["only_b"], c["both_absent"]) == (1, 1, 1, 1)

    def test_three_identities_yield_three_pairs(self) -> None:
        rows = [
            _row("t1", "cat_x", "a:1"),
            _row("t1", "cat_x", "b:2"),
            _row("t1", "cat_x", "c:3"),
        ]
        summary = compute_agreement(rows)
        names = {(p["annotator_a"], p["annotator_b"]) for p in summary["pairs"]}
        assert names == {("a:1", "b:2"), ("a:1", "c:3"), ("b:2", "c:3")}


# ---------------------------------------------------------------------------
# Tests — format_markdown
# ---------------------------------------------------------------------------


class TestFormatMarkdown:
    def test_contains_headers_and_values(self) -> None:
        rows = [
            _row("t1", "retrieval_failure", "heuristic:rule-engine"),
            _row("t2", "query_churn", "heuristic:rule-engine"),
            _row("t1", "retrieval_failure", "llm:haiku-4", "llm"),
            _row("t2", "query_churn", "llm:haiku-4", "llm"),
        ]
        md = format_markdown(compute_agreement(rows))

        assert "# Inter-Annotator Agreement" in md
        assert "heuristic:rule-engine vs llm:haiku-4" in md
        assert "retrieval_failure" in md
        assert "1.00" in md

    def test_undefined_kappa_rendered_na(self) -> None:
        rows = [
            _row("t1", "exception_crash", "a:1"),
            _row("t1", "exception_crash", "b:2"),
        ]
        md = format_markdown(compute_agreement(rows))
        assert "n/a" in md


# ---------------------------------------------------------------------------
# Tests — cmd_agreement CLI
# ---------------------------------------------------------------------------


class TestCmdAgreement:
    def _write_store(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_writes_markdown_and_json(self, tmp_path: Path) -> None:
        from agent_diagnostics.cli import cmd_agreement

        store = tmp_path / "annotations.jsonl"
        self._write_store(
            store,
            [
                _row("t1", "retrieval_failure", "heuristic:rule-engine"),
                _row("t1", "retrieval_failure", "llm:haiku-4", "llm"),
                _row("t2", "query_churn", "heuristic:rule-engine"),
                _row("t2", "query_churn", "llm:haiku-4", "llm"),
            ],
        )
        out_dir = tmp_path / "report"
        cmd_agreement(argparse.Namespace(annotations=str(store), output_dir=str(out_dir)))

        assert (out_dir / "agreement.md").is_file()
        summary = json.loads((out_dir / "agreement.json").read_text())
        assert summary["schema_version"] == SCHEMA_VERSION
        assert len(summary["pairs"]) == 1

    def test_single_annotator_warns(self, tmp_path: Path, caplog) -> None:
        from agent_diagnostics.cli import cmd_agreement

        store = tmp_path / "annotations.jsonl"
        self._write_store(store, [_row("t1", "retrieval_failure", "a:1")])
        out_dir = tmp_path / "report"

        with caplog.at_level(logging.WARNING, logger="agent_diagnostics.cli"):
            cmd_agreement(argparse.Namespace(annotations=str(store), output_dir=str(out_dir)))

        assert any("fewer than two annotator" in r.message for r in caplog.records)
        assert (out_dir / "agreement.json").is_file()

    def test_missing_file_exits(self, tmp_path: Path) -> None:
        from agent_diagnostics.cli import cmd_agreement

        with pytest.raises(SystemExit):
            cmd_agreement(
                argparse.Namespace(
                    annotations=str(tmp_path / "missing.jsonl"),
                    output_dir=str(tmp_path / "report"),
                )
            )
