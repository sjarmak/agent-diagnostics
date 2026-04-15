"""Tests for the Parquet export module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "query"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data_dir(tmp_path: Path) -> Path:
    """Create a data directory with test JSONL files and sidecar metadata."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # signals.jsonl — includes list columns
    signals = [
        {
            "trial_id": "aaa111",
            "task_id": "task-alpha-001",
            "model": "claude-sonnet-4-6",
            "agent_name": "claude-code",
            "config_name": "",
            "benchmark": "sdlc_secure",
            "reward": 0.6,
            "passed": True,
            "total_turns": 50,
            "tool_calls_total": 26,
            "error_count": 14,
            "duration_seconds": 431.8,
            "tool_call_sequence": ["Read", "Edit", "Bash"],
            "files_read_list": ["src/main.py", "tests/test_main.py"],
            "files_edited_list": ["src/main.py"],
        },
        {
            "trial_id": "bbb222",
            "task_id": "task-beta-002",
            "model": "claude-sonnet-4-6",
            "agent_name": "claude-code",
            "config_name": "",
            "benchmark": "sdlc_test",
            "reward": 1.0,
            "passed": True,
            "total_turns": 18,
            "tool_calls_total": 9,
            "error_count": 6,
            "duration_seconds": 132.5,
            "tool_call_sequence": ["Grep", "Read"],
            "files_read_list": ["README.md"],
            "files_edited_list": [],
        },
    ]
    signals_path = data_dir / "signals.jsonl"
    with open(signals_path, "w") as f:
        for row in signals:
            f.write(json.dumps(row) + "\n")

    # signals.meta.json sidecar
    meta = {
        "schema_version": "observatory-signals-v1",
        "taxonomy_version": "3.0",
        "generated_at": "2026-04-15T00:00:00+00:00",
    }
    with open(data_dir / "signals.meta.json", "w") as f:
        json.dump(meta, f)

    # annotations.jsonl
    annotations = [
        {
            "task_id": "task-alpha-001",
            "category": "premature_termination",
            "confidence": 0.85,
            "annotator": "heuristic",
        },
        {
            "task_id": "task-beta-002",
            "category": "clean_success",
            "confidence": 0.95,
            "annotator": "heuristic",
        },
    ]
    annotations_path = data_dir / "annotations.jsonl"
    with open(annotations_path, "w") as f:
        for row in annotations:
            f.write(json.dumps(row) + "\n")

    # annotations.meta.json sidecar
    ann_meta = {
        "schema_version": "observatory-annotation-v1",
        "taxonomy_version": "3.0",
        "generated_at": "2026-04-15T00:00:00+00:00",
    }
    with open(data_dir / "annotations.meta.json", "w") as f:
        json.dump(ann_meta, f)

    # manifests.jsonl
    manifests = [
        {
            "manifest_id": "manifest-001",
            "benchmark": "sdlc_secure",
            "task_count": 25,
            "created_at": "2026-03-10T12:00:00Z",
        },
    ]
    manifests_path = data_dir / "manifests.jsonl"
    with open(manifests_path, "w") as f:
        for row in manifests:
            f.write(json.dumps(row) + "\n")

    return data_dir


# ---------------------------------------------------------------------------
# export_parquet — core tests
# ---------------------------------------------------------------------------


class TestExportParquet:
    """Tests for export_parquet."""

    def test_produces_all_output_files(self, tmp_path):
        """Export creates signals.parquet, annotations.parquet, manifests.parquet, MANIFEST.json."""
        from agent_diagnostics.export import export_parquet

        data_dir = _make_data_dir(tmp_path)
        out_dir = tmp_path / "export"

        manifest = export_parquet(data_dir, out_dir)

        assert (out_dir / "signals.parquet").is_file()
        assert (out_dir / "annotations.parquet").is_file()
        assert (out_dir / "manifests.parquet").is_file()
        assert (out_dir / "MANIFEST.json").is_file()
        assert isinstance(manifest, dict)

    def test_parquet_uses_zstd_compression(self, tmp_path):
        """Parquet files use zstd compression."""
        from agent_diagnostics.export import export_parquet

        data_dir = _make_data_dir(tmp_path)
        out_dir = tmp_path / "export"
        export_parquet(data_dir, out_dir)

        for name in ("signals", "annotations", "manifests"):
            pq_path = out_dir / f"{name}.parquet"
            meta = pq.read_metadata(str(pq_path))
            for i in range(meta.num_row_groups):
                rg = meta.row_group(i)
                for j in range(rg.num_columns):
                    col = rg.column(j)
                    assert col.compression == "ZSTD", (
                        f"{name}.parquet column {j} uses "
                        f"{col.compression} instead of ZSTD"
                    )

    def test_list_columns_are_native_list_string(self, tmp_path):
        """signals.parquet list columns have pyarrow list<string> type."""
        from agent_diagnostics.export import export_parquet

        data_dir = _make_data_dir(tmp_path)
        out_dir = tmp_path / "export"
        export_parquet(data_dir, out_dir)

        table = pq.read_table(str(out_dir / "signals.parquet"))
        schema = table.schema

        for col_name in ("tool_call_sequence", "files_read_list", "files_edited_list"):
            field = schema.field(col_name)
            assert pa.types.is_list(
                field.type
            ), f"{col_name} should be list type, got {field.type}"
            assert pa.types.is_string(
                field.type.value_type
            ), f"{col_name} should have string value type, got {field.type.value_type}"

    def test_manifest_json_has_required_fields(self, tmp_path):
        """MANIFEST.json contains all required fields."""
        from agent_diagnostics.export import export_parquet

        data_dir = _make_data_dir(tmp_path)
        out_dir = tmp_path / "export"
        export_parquet(data_dir, out_dir)

        with open(out_dir / "MANIFEST.json") as f:
            manifest = json.load(f)

        required_keys = {
            "schema_version",
            "taxonomy_version",
            "row_count",
            "sha256_per_file",
            "source_commit",
            "generated_at",
        }
        assert required_keys <= set(manifest.keys())
        assert manifest["row_count"]["signals"] == 2
        assert manifest["row_count"]["annotations"] == 2
        assert manifest["row_count"]["manifests"] == 1
        assert len(manifest["sha256_per_file"]) == 3

    def test_manifest_validates_against_schema(self, tmp_path):
        """MANIFEST.json validates against export_manifest_schema.json."""
        import jsonschema

        from agent_diagnostics.export import export_parquet

        data_dir = _make_data_dir(tmp_path)
        out_dir = tmp_path / "export"
        export_parquet(data_dir, out_dir)

        schema_path = (
            Path(__file__).parent.parent
            / "src"
            / "agent_diagnostics"
            / "export_manifest_schema.json"
        )
        with open(schema_path) as f:
            schema = json.load(f)
        with open(out_dir / "MANIFEST.json") as f:
            manifest = json.load(f)

        # Should not raise
        jsonschema.validate(instance=manifest, schema=schema)

    def test_rerun_produces_byte_identical_parquet(self, tmp_path):
        """Re-running on unchanged inputs produces byte-identical Parquet."""
        from agent_diagnostics.export import export_parquet

        data_dir = _make_data_dir(tmp_path)
        out_dir_1 = tmp_path / "export1"
        out_dir_2 = tmp_path / "export2"

        export_parquet(data_dir, out_dir_1)
        export_parquet(data_dir, out_dir_2)

        for name in ("signals.parquet", "annotations.parquet", "manifests.parquet"):
            bytes_1 = (out_dir_1 / name).read_bytes()
            bytes_2 = (out_dir_2 / name).read_bytes()
            assert bytes_1 == bytes_2, f"{name} differs between runs"

    def test_pandas_read_parquet_works(self, tmp_path):
        """pandas.read_parquet on signals.parquet works with only pandas+pyarrow."""
        import pandas as pd

        from agent_diagnostics.export import export_parquet

        data_dir = _make_data_dir(tmp_path)
        out_dir = tmp_path / "export"
        export_parquet(data_dir, out_dir)

        df = pd.read_parquet(out_dir / "signals.parquet")
        assert len(df) == 2
        assert "tool_call_sequence" in df.columns
        # List columns should be iterable sequences when read back
        first_tcs = df["tool_call_sequence"].iloc[0]
        # pandas may return a numpy array or list depending on version
        assert hasattr(first_tcs, "__iter__")
        assert list(first_tcs) == ["Read", "Edit", "Bash"]

    def test_missing_data_dir_raises(self, tmp_path):
        """export_parquet raises FileNotFoundError for missing data_dir."""
        from agent_diagnostics.export import export_parquet

        with pytest.raises(FileNotFoundError):
            export_parquet(tmp_path / "nonexistent", tmp_path / "out")

    def test_empty_jsonl_files(self, tmp_path):
        """Export handles empty or missing JSONL files gracefully."""
        from agent_diagnostics.export import export_parquet

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Create empty signals.jsonl
        (data_dir / "signals.jsonl").write_text("")
        out_dir = tmp_path / "export"

        manifest = export_parquet(data_dir, out_dir)
        assert manifest["row_count"]["signals"] == 0
        assert manifest["row_count"]["annotations"] == 0

    def test_multiple_taxonomy_versions_raises(self, tmp_path):
        """export_parquet raises ValueError when taxonomy versions conflict."""
        from agent_diagnostics.export import export_parquet

        data_dir = _make_data_dir(tmp_path)

        # Override annotations sidecar with different taxonomy_version
        ann_meta = {
            "schema_version": "observatory-annotation-v1",
            "taxonomy_version": "2.0",
            "generated_at": "2026-04-15T00:00:00+00:00",
        }
        with open(data_dir / "annotations.meta.json", "w") as f:
            json.dump(ann_meta, f)

        out_dir = tmp_path / "export"
        with pytest.raises(ValueError, match="Multiple taxonomy versions"):
            export_parquet(data_dir, out_dir)

    def test_schema_version_detected_from_sidecar(self, tmp_path):
        """schema_version and taxonomy_version are read from sidecars."""
        from agent_diagnostics.export import export_parquet

        data_dir = _make_data_dir(tmp_path)
        out_dir = tmp_path / "export"
        manifest = export_parquet(data_dir, out_dir)

        assert manifest["schema_version"] == "observatory-signals-v1"
        assert manifest["taxonomy_version"] == "3.0"

    def test_rows_sorted_deterministically(self, tmp_path):
        """Rows are sorted by trial_id for signals."""
        from agent_diagnostics.export import export_parquet

        data_dir = _make_data_dir(tmp_path)
        out_dir = tmp_path / "export"
        export_parquet(data_dir, out_dir)

        table = pq.read_table(str(out_dir / "signals.parquet"))
        trial_ids = table.column("trial_id").to_pylist()
        assert trial_ids == sorted(trial_ids)


# ---------------------------------------------------------------------------
# refresh_manifests
# ---------------------------------------------------------------------------


class TestRefreshManifests:
    """Tests for refresh_manifests."""

    def test_refresh_creates_manifests_jsonl(self, tmp_path):
        """refresh_manifests rewrites manifests.jsonl from signals."""
        from agent_diagnostics.export import refresh_manifests

        data_dir = _make_data_dir(tmp_path)
        output_path = refresh_manifests(data_dir)

        assert output_path.is_file()
        lines = output_path.read_text().strip().splitlines()
        rows = [json.loads(line) for line in lines]
        # Should have 2 benchmarks: sdlc_secure and sdlc_test
        benchmarks = {r["benchmark"] for r in rows}
        assert "sdlc_secure" in benchmarks
        assert "sdlc_test" in benchmarks


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCmdExport:
    """Tests for cmd_export CLI command."""

    def test_cmd_export_success(self, tmp_path):
        """cmd_export produces Parquet files and MANIFEST.json."""
        from agent_diagnostics.cli import cmd_export

        data_dir = _make_data_dir(tmp_path)
        out_dir = tmp_path / "export"

        args = argparse.Namespace(
            data_dir=str(data_dir),
            format="parquet",
            out=str(out_dir),
        )
        cmd_export(args)

        assert (out_dir / "signals.parquet").is_file()
        assert (out_dir / "MANIFEST.json").is_file()

    def test_cmd_export_missing_data_dir(self, tmp_path):
        """cmd_export exits with error for missing data dir."""
        from agent_diagnostics.cli import cmd_export

        args = argparse.Namespace(
            data_dir=str(tmp_path / "nonexistent"),
            format="parquet",
            out=str(tmp_path / "export"),
        )
        with pytest.raises(SystemExit):
            cmd_export(args)

    def test_cmd_export_unsupported_format(self, tmp_path):
        """cmd_export exits with error for unsupported format."""
        from agent_diagnostics.cli import cmd_export

        data_dir = _make_data_dir(tmp_path)
        args = argparse.Namespace(
            data_dir=str(data_dir),
            format="csv",
            out=str(tmp_path / "export"),
        )
        with pytest.raises(SystemExit):
            cmd_export(args)


class TestCmdManifestRefresh:
    """Tests for cmd_manifest_refresh CLI command."""

    def test_cmd_manifest_refresh_success(self, tmp_path):
        """cmd_manifest_refresh rewrites manifests.jsonl."""
        from agent_diagnostics.cli import cmd_manifest_refresh

        data_dir = _make_data_dir(tmp_path)
        args = argparse.Namespace(data_dir=str(data_dir))
        cmd_manifest_refresh(args)

        manifests_path = data_dir / "manifests.jsonl"
        assert manifests_path.is_file()

    def test_cmd_manifest_refresh_missing_dir(self, tmp_path):
        """cmd_manifest_refresh exits with error for missing dir."""
        from agent_diagnostics.cli import cmd_manifest_refresh

        args = argparse.Namespace(data_dir=str(tmp_path / "nonexistent"))
        with pytest.raises(SystemExit):
            cmd_manifest_refresh(args)


# ---------------------------------------------------------------------------
# Fixture-based integration test
# ---------------------------------------------------------------------------


class TestFixtureExport:
    """Integration test using the existing query fixtures."""

    def test_export_from_fixtures(self, tmp_path):
        """Export from the query fixture directory works end-to-end."""
        from agent_diagnostics.export import export_parquet

        out_dir = tmp_path / "export"
        manifest = export_parquet(FIXTURES_DIR, out_dir)

        assert manifest["row_count"]["signals"] == 5
        assert manifest["row_count"]["annotations"] == 5
        assert manifest["row_count"]["manifests"] == 2
        assert (out_dir / "signals.parquet").is_file()
        assert (out_dir / "MANIFEST.json").is_file()
