"""Tests for agent_diagnostics.model_identity."""

from __future__ import annotations

import pytest

from agent_diagnostics.model_identity import (
    load_models_config,
    resolve_identity,
    resolve_snapshot,
)

# ---------------------------------------------------------------------------
# load_models_config
# ---------------------------------------------------------------------------


class TestLoadModelsConfig:
    def test_returns_dict_with_models_key(self) -> None:
        config = load_models_config()
        assert isinstance(config, dict)
        assert "models" in config

    def test_models_contains_expected_identities(self) -> None:
        config = load_models_config()
        models = config["models"]
        for identity in ("llm:haiku-4", "llm:sonnet-4", "llm:opus-4"):
            assert identity in models, f"Missing identity {identity!r}"

    def test_each_identity_has_snapshot_ids(self) -> None:
        config = load_models_config()
        for logical_id, entry in config["models"].items():
            assert "snapshot_ids" in entry, f"{logical_id} missing snapshot_ids"
            assert len(entry["snapshot_ids"]) > 0, f"{logical_id} has empty snapshot_ids"


# ---------------------------------------------------------------------------
# resolve_identity
# ---------------------------------------------------------------------------


class TestResolveIdentity:
    def test_haiku_snapshot_returns_haiku_identity(self) -> None:
        assert resolve_identity("claude-haiku-4-5-20251001") == "llm:haiku-4"

    def test_sonnet_snapshots_return_sonnet_identity(self) -> None:
        assert resolve_identity("claude-sonnet-4-5-20250514") == "llm:sonnet-4"
        assert resolve_identity("claude-sonnet-4-6-20260514") == "llm:sonnet-4"

    def test_opus_snapshots_return_opus_identity(self) -> None:
        assert resolve_identity("claude-opus-4-5-20250514") == "llm:opus-4"
        assert resolve_identity("claude-opus-4-6-20260514") == "llm:opus-4"

    def test_unknown_snapshot_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown snapshot ID"):
            resolve_identity("claude-nonexistent-99-20301231")

    def test_error_message_lists_known_ids(self) -> None:
        with pytest.raises(ValueError, match="claude-haiku-4-5-20251001"):
            resolve_identity("not-a-real-model")


# ---------------------------------------------------------------------------
# resolve_snapshot
# ---------------------------------------------------------------------------


class TestResolveSnapshot:
    def test_haiku_returns_first_snapshot(self) -> None:
        assert resolve_snapshot("llm:haiku-4") == "claude-haiku-4-5-20251001"

    def test_sonnet_returns_first_snapshot(self) -> None:
        # First listed snapshot is the "current" one
        assert resolve_snapshot("llm:sonnet-4") == "claude-sonnet-4-6-20260514"

    def test_opus_returns_first_snapshot(self) -> None:
        assert resolve_snapshot("llm:opus-4") == "claude-opus-4-6-20260514"

    def test_unknown_identity_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown logical identity"):
            resolve_snapshot("llm:nonexistent-99")

    def test_error_message_lists_known_identities(self) -> None:
        with pytest.raises(ValueError, match="llm:haiku-4"):
            resolve_snapshot("llm:nonexistent-99")


# ---------------------------------------------------------------------------
# Round-trip consistency
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Verify that resolve_identity and resolve_snapshot are consistent."""

    @pytest.mark.parametrize(
        "logical_id",
        ["llm:haiku-4", "llm:sonnet-4", "llm:opus-4"],
    )
    def test_snapshot_round_trips_to_same_identity(self, logical_id: str) -> None:
        snapshot = resolve_snapshot(logical_id)
        assert resolve_identity(snapshot) == logical_id
