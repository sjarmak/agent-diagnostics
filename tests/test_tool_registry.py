"""Tests for the tool registry module."""

from dataclasses import FrozenInstanceError

import pytest

from agent_diagnostics.tool_registry import DEFAULT_REGISTRY, ToolRegistry


class TestToolRegistryDataclass:
    """Verify ToolRegistry is a frozen dataclass with correct fields."""

    def test_is_frozen(self) -> None:
        registry = ToolRegistry(
            search_tools=frozenset({"a"}),
            edit_tools=frozenset({"b"}),
            code_nav_tools=frozenset({"c"}),
            semantic_search_tools=frozenset({"d"}),
        )
        with pytest.raises(FrozenInstanceError):
            registry.search_tools = frozenset({"x"})  # type: ignore[misc]

    def test_custom_creation(self) -> None:
        registry = ToolRegistry(
            search_tools=frozenset({"my_grep"}),
            edit_tools=frozenset({"my_edit"}),
            code_nav_tools=frozenset({"my_nav"}),
            semantic_search_tools=frozenset({"my_search"}),
        )
        assert registry.search_tools == frozenset({"my_grep"})
        assert registry.edit_tools == frozenset({"my_edit"})
        assert registry.code_nav_tools == frozenset({"my_nav"})
        assert registry.semantic_search_tools == frozenset({"my_search"})

    def test_fields_are_frozensets(self) -> None:
        registry = ToolRegistry(
            search_tools=frozenset(),
            edit_tools=frozenset(),
            code_nav_tools=frozenset(),
            semantic_search_tools=frozenset(),
        )
        assert isinstance(registry.search_tools, frozenset)
        assert isinstance(registry.edit_tools, frozenset)
        assert isinstance(registry.code_nav_tools, frozenset)
        assert isinstance(registry.semantic_search_tools, frozenset)


class TestAllToolsProperty:
    """Verify the all_tools property returns the union of all sets."""

    def test_all_tools_union(self) -> None:
        registry = ToolRegistry(
            search_tools=frozenset({"a", "b"}),
            edit_tools=frozenset({"c"}),
            code_nav_tools=frozenset({"d", "e"}),
            semantic_search_tools=frozenset({"f"}),
        )
        assert registry.all_tools == frozenset({"a", "b", "c", "d", "e", "f"})

    def test_all_tools_returns_frozenset(self) -> None:
        registry = ToolRegistry(
            search_tools=frozenset(),
            edit_tools=frozenset(),
            code_nav_tools=frozenset(),
            semantic_search_tools=frozenset(),
        )
        assert isinstance(registry.all_tools, frozenset)

    def test_all_tools_with_overlapping_names(self) -> None:
        registry = ToolRegistry(
            search_tools=frozenset({"shared"}),
            edit_tools=frozenset({"shared"}),
            code_nav_tools=frozenset({"unique"}),
            semantic_search_tools=frozenset(),
        )
        assert registry.all_tools == frozenset({"shared", "unique"})


class TestDefaultRegistry:
    """Verify DEFAULT_REGISTRY contains the expected tool names."""

    def test_search_tools_minimum(self) -> None:
        expected = {"Grep", "Glob", "mcp__sourcegraph__keyword_search"}
        assert expected.issubset(DEFAULT_REGISTRY.search_tools)

    def test_edit_tools_minimum(self) -> None:
        expected = {"Edit", "Write"}
        assert expected.issubset(DEFAULT_REGISTRY.edit_tools)

    def test_code_nav_tools_minimum(self) -> None:
        expected = {
            "Read",
            "mcp__sourcegraph__read_file",
            "mcp__sourcegraph__go_to_definition",
        }
        assert expected.issubset(DEFAULT_REGISTRY.code_nav_tools)

    def test_semantic_search_tools_minimum(self) -> None:
        expected = {"mcp__sourcegraph__nls_search"}
        assert expected.issubset(DEFAULT_REGISTRY.semantic_search_tools)

    def test_all_tools_covers_every_set(self) -> None:
        all_tools = DEFAULT_REGISTRY.all_tools
        assert DEFAULT_REGISTRY.search_tools.issubset(all_tools)
        assert DEFAULT_REGISTRY.edit_tools.issubset(all_tools)
        assert DEFAULT_REGISTRY.code_nav_tools.issubset(all_tools)
        assert DEFAULT_REGISTRY.semantic_search_tools.issubset(all_tools)

    def test_is_tool_registry_instance(self) -> None:
        assert isinstance(DEFAULT_REGISTRY, ToolRegistry)


class TestImport:
    """Verify the public import path works."""

    def test_import_from_module(self) -> None:
        from agent_diagnostics.tool_registry import DEFAULT_REGISTRY, ToolRegistry

        assert ToolRegistry is not None
        assert DEFAULT_REGISTRY is not None
