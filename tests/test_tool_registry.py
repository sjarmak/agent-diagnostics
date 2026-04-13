"""Tests for the tool registry module."""

from dataclasses import FrozenInstanceError

import pytest

from agent_diagnostics.tool_registry import (
    DEFAULT_REGISTRY,
    OPENHANDS_REGISTRY,
    ToolRegistry,
    get_registry_for_agent,
)


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


class TestOpenHandsRegistry:
    """Verify OPENHANDS_REGISTRY contains the expected tool mappings."""

    def test_is_tool_registry_instance(self) -> None:
        assert isinstance(OPENHANDS_REGISTRY, ToolRegistry)

    def test_str_replace_editor_in_edit_tools(self) -> None:
        assert "str_replace_editor" in OPENHANDS_REGISTRY.edit_tools

    def test_execute_bash_in_search_tools(self) -> None:
        assert "execute_bash" in OPENHANDS_REGISTRY.search_tools

    def test_browser_in_code_nav_tools(self) -> None:
        assert "browser" in OPENHANDS_REGISTRY.code_nav_tools

    def test_semantic_search_tools_empty(self) -> None:
        assert OPENHANDS_REGISTRY.semantic_search_tools == frozenset()

    def test_all_tools_covers_every_set(self) -> None:
        all_tools = OPENHANDS_REGISTRY.all_tools
        assert OPENHANDS_REGISTRY.search_tools.issubset(all_tools)
        assert OPENHANDS_REGISTRY.edit_tools.issubset(all_tools)
        assert OPENHANDS_REGISTRY.code_nav_tools.issubset(all_tools)

    def test_is_distinct_from_default(self) -> None:
        assert OPENHANDS_REGISTRY is not DEFAULT_REGISTRY


class TestGetRegistryForAgent:
    """Verify agent-based registry selection."""

    def test_openhands_returns_openhands_registry(self) -> None:
        assert get_registry_for_agent("openhands") is OPENHANDS_REGISTRY

    def test_openhands_case_insensitive(self) -> None:
        assert get_registry_for_agent("OpenHands") is OPENHANDS_REGISTRY

    def test_openhands_as_substring(self) -> None:
        assert get_registry_for_agent("openhands-v2.0") is OPENHANDS_REGISTRY

    def test_openhands_mixed_case_substring(self) -> None:
        assert get_registry_for_agent("MyOpenHandsAgent") is OPENHANDS_REGISTRY

    def test_claude_code_returns_default(self) -> None:
        assert get_registry_for_agent("claude-code") is DEFAULT_REGISTRY

    def test_unknown_agent_returns_default(self) -> None:
        assert get_registry_for_agent("some-other-agent") is DEFAULT_REGISTRY

    def test_empty_string_returns_default(self) -> None:
        assert get_registry_for_agent("") is DEFAULT_REGISTRY


class TestImport:
    """Verify the public import path works."""

    def test_import_from_module(self) -> None:
        from agent_diagnostics.tool_registry import DEFAULT_REGISTRY, ToolRegistry

        assert ToolRegistry is not None
        assert DEFAULT_REGISTRY is not None

    def test_import_openhands_registry(self) -> None:
        from agent_diagnostics.tool_registry import OPENHANDS_REGISTRY

        assert OPENHANDS_REGISTRY is not None

    def test_import_get_registry_for_agent(self) -> None:
        from agent_diagnostics.tool_registry import get_registry_for_agent

        assert callable(get_registry_for_agent)
