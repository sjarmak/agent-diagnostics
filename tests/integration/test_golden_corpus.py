"""Integration tests for the golden regression corpus.

These tests verify that the hand-curated corpus under
``tests/fixtures/golden_corpus/`` is structurally valid, can be
round-tripped through the heuristic annotator and the FakeLLMBackend,
and is usable as a regression baseline.

Per-category recall is computed and reported (stdout/log) but NOT
asserted — calibration thresholds are the job of the downstream
calibration bead.  The goal here is to detect *structural* regressions:
missing files, malformed JSON, unknown categories, or catastrophic
breakage in ``annotate_trial``.

Runtime target: ≤ 30 seconds total.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, cast

import pytest

from agent_diagnostics.annotator import annotate_trial
from agent_diagnostics.taxonomy import valid_category_names
from agent_diagnostics.types import TrialSignals

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "golden_corpus"
TAXONOMY_PATH = Path(str(resources.files("agent_diagnostics") / "taxonomy_v3.yaml"))

# Acceptance criteria from bead agent-diagnostics-n0n
MIN_TRIALS = 30
MIN_AGENTS = 3
MIN_BENCHMARKS = 5
MAX_RUNTIME_SECONDS = 30.0
MIN_KAPPA = 0.6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenTrial:
    """Loaded golden-corpus trial."""

    trial_id_short: str
    signals: dict[str, Any]
    trajectory: dict[str, Any]
    metadata: dict[str, Any]
    expected: dict[str, Any]


def _load_trial(trial_dir: Path) -> GoldenTrial:
    return GoldenTrial(
        trial_id_short=trial_dir.name,
        signals=json.loads((trial_dir / "signals.json").read_text(encoding="utf-8")),
        trajectory=json.loads(
            (trial_dir / "trajectory.json").read_text(encoding="utf-8")
        ),
        metadata=json.loads((trial_dir / "metadata.json").read_text(encoding="utf-8")),
        expected=json.loads(
            (trial_dir / "expected_annotations.json").read_text(encoding="utf-8")
        ),
    )


@pytest.fixture(scope="module")
def golden_trials() -> list[GoldenTrial]:
    """Load all trials from the golden corpus directory."""
    trial_dirs = sorted(d for d in CORPUS_DIR.iterdir() if d.is_dir())
    trials = [_load_trial(d) for d in trial_dirs]
    assert trials, f"golden corpus empty: {CORPUS_DIR}"
    return trials


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    """Load the corpus manifest."""
    path = CORPUS_DIR / "MANIFEST.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def valid_v3_category_names() -> frozenset[str]:
    return frozenset(valid_category_names(TAXONOMY_PATH))


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


class TestCorpusStructure:
    """Verify fixture file structure and manifest coverage."""

    def test_corpus_dir_exists(self) -> None:
        assert CORPUS_DIR.is_dir(), f"corpus dir missing: {CORPUS_DIR}"

    def test_manifest_exists(self) -> None:
        assert (CORPUS_DIR / "MANIFEST.json").is_file()

    def test_readme_exists(self) -> None:
        assert (CORPUS_DIR / "README.md").is_file()

    def test_minimum_trial_count(self, golden_trials: list[GoldenTrial]) -> None:
        assert (
            len(golden_trials) >= MIN_TRIALS
        ), f"expected >= {MIN_TRIALS} trials, got {len(golden_trials)}"

    def test_each_trial_has_all_files(self) -> None:
        expected_files = {
            "signals.json",
            "trajectory.json",
            "metadata.json",
            "expected_annotations.json",
        }
        for trial_dir in CORPUS_DIR.iterdir():
            if not trial_dir.is_dir():
                continue
            files_present = {p.name for p in trial_dir.iterdir() if p.is_file()}
            missing = expected_files - files_present
            assert not missing, f"trial {trial_dir.name} missing {missing}"


class TestCoverageMatrix:
    """Verify selection criteria are met."""

    def test_at_least_three_agents(self, manifest: dict[str, Any]) -> None:
        agents = manifest.get("agents", [])
        assert (
            len(agents) >= MIN_AGENTS
        ), f"expected >= {MIN_AGENTS} agents, got {agents}"

    def test_at_least_five_benchmarks(self, manifest: dict[str, Any]) -> None:
        benchmarks = manifest.get("benchmarks", [])
        assert (
            len(benchmarks) >= MIN_BENCHMARKS
        ), f"expected >= {MIN_BENCHMARKS} benchmarks, got {benchmarks}"

    def test_mix_of_passed_and_failed(self, manifest: dict[str, Any]) -> None:
        passed = manifest.get("passed_count", 0)
        failed = manifest.get("failed_count", 0)
        assert passed > 0 and failed > 0
        # Rough 50/50 check: neither side exceeds 80% of the total.
        total = passed + failed
        assert passed / total < 0.8, f"passed dominates: {passed}/{total}"
        assert failed / total < 0.8, f"failed dominates: {failed}/{total}"

    def test_kappa_meets_minimum(self, manifest: dict[str, Any]) -> None:
        kappa = manifest.get("cohen_kappa")
        assert kappa is not None, "manifest missing cohen_kappa"
        assert (
            kappa >= MIN_KAPPA
        ), f"kappa {kappa:.3f} below target {MIN_KAPPA}"


class TestCategoryValidity:
    """Verify every expected category is a known v3 taxonomy name."""

    def test_expected_categories_in_taxonomy(
        self,
        golden_trials: list[GoldenTrial],
        valid_v3_category_names: frozenset[str],
    ) -> None:
        for trial in golden_trials:
            for cat in trial.expected.get("categories", []):
                name = cat.get("name")
                assert (
                    name in valid_v3_category_names
                ), f"trial {trial.trial_id_short}: unknown category {name}"

    def test_expected_confidence_in_range(
        self, golden_trials: list[GoldenTrial]
    ) -> None:
        for trial in golden_trials:
            for cat in trial.expected.get("categories", []):
                conf = cat.get("confidence")
                assert (
                    conf is None or 0.0 <= conf <= 1.0
                ), f"{trial.trial_id_short}: confidence out of range {conf}"

    def test_curator_notes_present(self, golden_trials: list[GoldenTrial]) -> None:
        for trial in golden_trials:
            notes = trial.expected.get("curator_notes") or {}
            assert notes.get("methodology"), (
                f"{trial.trial_id_short}: curator_notes.methodology missing"
            )
            assert notes.get("curator_model"), (
                f"{trial.trial_id_short}: curator_notes.curator_model missing"
            )


# ---------------------------------------------------------------------------
# Heuristic annotator round-trip
# ---------------------------------------------------------------------------


class TestHeuristicAnnotator:
    """Run :func:`annotate_trial` on every trial and assert no crash."""

    def test_annotate_each_trial_smoke(
        self, golden_trials: list[GoldenTrial]
    ) -> None:
        for trial in golden_trials:
            result = annotate_trial(cast(TrialSignals, trial.signals))
            assert isinstance(result, list)

    def test_annotate_within_runtime_budget(
        self, golden_trials: list[GoldenTrial]
    ) -> None:
        start = time.perf_counter()
        for trial in golden_trials:
            annotate_trial(cast(TrialSignals, trial.signals))
        elapsed = time.perf_counter() - start
        assert (
            elapsed < MAX_RUNTIME_SECONDS
        ), f"annotate_trial x {len(golden_trials)} took {elapsed:.2f}s"

    def test_heuristic_recall_report(
        self, golden_trials: list[GoldenTrial]
    ) -> None:
        """Compute per-category recall and report it.

        Reports, does NOT assert thresholds.  Calibration bead owns
        recall gating.
        """
        per_cat_expected: Counter[str] = Counter()
        per_cat_hits: Counter[str] = Counter()

        for trial in golden_trials:
            expected_names = {
                c["name"] for c in trial.expected.get("categories", [])
            }
            heuristic_result = annotate_trial(cast(TrialSignals, trial.signals))
            predicted_names = {a.name for a in heuristic_result}
            for cat in expected_names:
                per_cat_expected[cat] += 1
                if cat in predicted_names:
                    per_cat_hits[cat] += 1

        print("\n=== Heuristic per-category recall (report only) ===")
        for cat in sorted(per_cat_expected):
            expected = per_cat_expected[cat]
            hits = per_cat_hits[cat]
            recall = hits / expected if expected else float("nan")
            print(f"  {cat:32s}  expected={expected:3d}  hits={hits:3d}  recall={recall:.2f}")

        # Sanity: at least one category has non-zero recall
        assert any(per_cat_hits.values()), "heuristic annotator produced zero hits"


# ---------------------------------------------------------------------------
# FakeLLMBackend round-trip
# ---------------------------------------------------------------------------


class _SeededFakeBackend:
    """Deterministic per-trial fake backend.

    Unlike the general ``FakeLLMBackend`` which matches categories by
    keyword, this backend is seeded at construction time with the
    expected annotations for every trial, keyed by ``trial_id_short``.
    Looks up the trial id embedded in each prompt and returns the
    expected categories verbatim.
    """

    def __init__(self, seed: dict[str, list[dict[str, Any]]]) -> None:
        self._seed = seed
        self.call_count = 0

    def annotate(self, prompt: str) -> dict[str, Any]:
        self.call_count += 1
        for trial_id, cats in self._seed.items():
            if trial_id in prompt:
                return {"categories": list(cats)}
        return {"categories": []}


def _build_prompt(trial: GoldenTrial) -> str:
    return (
        f"Annotate trial {trial.trial_id_short} with v3 taxonomy categories.\n"
        f"Metadata: {json.dumps(trial.metadata)}"
    )


class TestLLMRoundTrip:
    """Verify a seeded fake LLM reproduces expected annotations."""

    def test_fake_roundtrip_matches_expected(
        self, golden_trials: list[GoldenTrial]
    ) -> None:
        seed = {
            t.trial_id_short: t.expected.get("categories", []) for t in golden_trials
        }
        backend = _SeededFakeBackend(seed)

        for trial in golden_trials:
            response = backend.annotate(_build_prompt(trial))
            returned_names = {c["name"] for c in response["categories"]}
            expected_names = {
                c["name"] for c in trial.expected.get("categories", [])
            }
            assert returned_names == expected_names, (
                f"{trial.trial_id_short}: round-trip mismatch "
                f"{returned_names ^ expected_names}"
            )

        assert backend.call_count == len(golden_trials)


# ---------------------------------------------------------------------------
# Overall runtime guard
# ---------------------------------------------------------------------------


def test_overall_runtime_budget(golden_trials: list[GoldenTrial]) -> None:
    """Run the heuristic + LLM round-trip end-to-end within 30s."""
    start = time.perf_counter()

    # Heuristic pass
    for trial in golden_trials:
        annotate_trial(cast(TrialSignals, trial.signals))

    # Fake LLM pass
    seed = {t.trial_id_short: t.expected.get("categories", []) for t in golden_trials}
    backend = _SeededFakeBackend(seed)
    for trial in golden_trials:
        backend.annotate(_build_prompt(trial))

    elapsed = time.perf_counter() - start
    assert (
        elapsed < MAX_RUNTIME_SECONDS
    ), f"end-to-end runtime {elapsed:.2f}s exceeds {MAX_RUNTIME_SECONDS}s"
