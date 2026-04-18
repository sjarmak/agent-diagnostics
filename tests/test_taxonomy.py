"""Tests for the agent-observatory taxonomy package."""

import json
from pathlib import Path

import pytest

import agent_diagnostics
from agent_diagnostics.taxonomy import (
    _extract_categories,
    _is_v2,
    _package_data_path,
    get_schema_path,
    load_taxonomy,
    valid_category_names,
    validate_annotation_categories,
)


class TestPackageDataResolution:
    """Verify bundled data files are accessible via importlib.resources."""

    def test_taxonomy_v1_exists(self) -> None:
        path = _package_data_path("taxonomy_v1.yaml")
        assert path.exists(), f"taxonomy_v1.yaml not found at {path}"

    def test_taxonomy_v2_exists(self) -> None:
        path = _package_data_path("taxonomy_v2.yaml")
        assert path.exists(), f"taxonomy_v2.yaml not found at {path}"

    def test_annotation_schema_exists(self) -> None:
        path = get_schema_path()
        assert path.exists(), f"annotation_schema.json not found at {path}"

    def test_schema_is_valid_json(self) -> None:
        path = get_schema_path()
        data = json.loads(path.read_text())
        assert data["$id"] == "observatory-annotation-v1"


class TestTaxonomyDefault:
    """Tests for the default (v3) taxonomy loaded by load_taxonomy()."""

    def test_load_default_taxonomy(self) -> None:
        taxonomy = load_taxonomy()
        assert taxonomy.get("version", "").startswith("3.")
        assert "dimensions" in taxonomy

    def test_default_is_v2_structured(self) -> None:
        taxonomy = load_taxonomy()
        assert _is_v2(taxonomy)


class TestTaxonomyV1:
    """Tests for v1 format taxonomy (loaded by explicit path)."""

    @staticmethod
    def _load_v1() -> dict:
        return load_taxonomy(_package_data_path("taxonomy_v1.yaml"))

    def test_load_v1(self) -> None:
        taxonomy = self._load_v1()
        assert "version" in taxonomy
        assert "categories" in taxonomy

    def test_v1_has_23_categories(self) -> None:
        taxonomy = self._load_v1()
        assert len(taxonomy["categories"]) == 23

    def test_v1_is_not_v2(self) -> None:
        taxonomy = self._load_v1()
        assert not _is_v2(taxonomy)

    def test_v1_category_required_fields(self) -> None:
        taxonomy = self._load_v1()
        required = {"name", "description", "polarity"}
        for cat in taxonomy["categories"]:
            missing = required - set(cat.keys())
            assert not missing, f"Category {cat.get('name', '?')} missing: {missing}"

    def test_v1_polarities(self) -> None:
        taxonomy = self._load_v1()
        valid_polarities = {"failure", "success", "neutral"}
        for cat in taxonomy["categories"]:
            assert (
                cat["polarity"] in valid_polarities
            ), f"{cat['name']} has invalid polarity: {cat['polarity']}"

    def test_v1_polarity_counts(self) -> None:
        taxonomy = self._load_v1()
        counts: dict[str, int] = {}
        for cat in taxonomy["categories"]:
            counts[cat["polarity"]] = counts.get(cat["polarity"], 0) + 1
        assert counts["failure"] >= 14
        assert counts["success"] >= 5
        assert counts["neutral"] >= 2


class TestTaxonomyV2:
    """Tests for v2 (dimensions) format taxonomy."""

    def test_load_v2(self) -> None:
        path = _package_data_path("taxonomy_v2.yaml")
        taxonomy = load_taxonomy(path)
        assert _is_v2(taxonomy)
        assert "dimensions" in taxonomy

    def test_v2_extract_categories_matches_v1(self) -> None:
        v1 = load_taxonomy(_package_data_path("taxonomy_v1.yaml"))
        path_v2 = _package_data_path("taxonomy_v2.yaml")
        v2 = load_taxonomy(path_v2)
        v1_names = {cat["name"] for cat in v1["categories"]}
        v2_names = {cat["name"] for cat in _extract_categories(v2)}
        assert (
            v1_names == v2_names
        ), f"Mismatch: {v1_names.symmetric_difference(v2_names)}"

    def test_v2_dimensions_have_categories(self) -> None:
        path = _package_data_path("taxonomy_v2.yaml")
        taxonomy = load_taxonomy(path)
        for dim in taxonomy["dimensions"]:
            assert "name" in dim
            assert "categories" in dim
            assert len(dim["categories"]) > 0


class TestValidCategoryNames:
    """Tests for valid_category_names helper."""

    def test_returns_set(self) -> None:
        names = valid_category_names()
        assert isinstance(names, set)
        # v3 taxonomy has 40 categories across 11 dimensions.
        assert len(names) == 40

    def test_known_categories_present(self) -> None:
        names = valid_category_names()
        expected = {
            "retrieval_failure",
            "query_churn",
            "success_via_code_nav",
            "rate_limited_run",
            "task_ambiguity",
        }
        assert expected.issubset(names)


class TestValidateAnnotation:
    """Tests for annotation validation."""

    def test_valid_annotation_passes(self) -> None:
        annotation = {
            "categories": [
                {"name": "retrieval_failure", "confidence": 0.9},
            ]
        }
        validate_annotation_categories(annotation)  # should not raise

    def test_invalid_category_raises(self) -> None:
        annotation = {
            "categories": [
                {"name": "not_a_real_category", "confidence": 0.5},
            ]
        }
        with pytest.raises(ValueError, match="not_a_real_category"):
            validate_annotation_categories(annotation)

    def test_empty_categories_passes(self) -> None:
        validate_annotation_categories({"categories": []})


class TestExemplars:
    """Validate that bundled exemplar files conform to expectations."""

    @pytest.fixture()
    def exemplar_dir(self) -> Path:
        return _package_data_path("exemplars")

    def test_exemplar_dir_exists(self, exemplar_dir: Path) -> None:
        assert exemplar_dir.is_dir()

    def test_25_exemplar_files(self, exemplar_dir: Path) -> None:
        jsons = list(exemplar_dir.glob("*.json"))
        assert len(jsons) == 25, f"Expected 25 exemplars, found {len(jsons)}"

    def test_exemplars_have_valid_categories(self, exemplar_dir: Path) -> None:
        valid = valid_category_names()
        for f in sorted(exemplar_dir.glob("*.json")):
            data = json.loads(f.read_text())
            for ann in data.get("annotations", []):
                for cat in ann.get("categories", []):
                    assert (
                        cat["name"] in valid
                    ), f"{f.name}: unknown category '{cat['name']}'"


class TestVersion:
    """Package version is accessible."""

    def test_version_string(self) -> None:
        assert agent_diagnostics.__version__ == "0.7.0"
