"""Injectable tool registry for classifying agent tool usage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolRegistry:
    """Immutable registry mapping tool names to capability categories.

    Each field holds a frozenset of tool name strings. External consumers
    can create custom registries for non-default agent harnesses.
    """

    search_tools: frozenset[str]
    edit_tools: frozenset[str]
    code_nav_tools: frozenset[str]
    semantic_search_tools: frozenset[str]

    @property
    def all_tools(self) -> frozenset[str]:
        """Return the union of all tool sets."""
        return (
            self.search_tools | self.edit_tools | self.code_nav_tools | self.semantic_search_tools
        )


DEFAULT_REGISTRY = ToolRegistry(
    search_tools=frozenset(
        {
            "Grep",
            "Glob",
            "mcp__sourcegraph__keyword_search",
            "mcp__sourcegraph__diff_search",
            "mcp__sourcegraph__commit_search",
        }
    ),
    edit_tools=frozenset(
        {
            "Edit",
            "Write",
            "NotebookEdit",
        }
    ),
    code_nav_tools=frozenset(
        {
            "Read",
            "mcp__sourcegraph__read_file",
            "mcp__sourcegraph__go_to_definition",
            "mcp__sourcegraph__find_references",
            "mcp__sourcegraph__list_files",
            "mcp__sourcegraph__compare_revisions",
        }
    ),
    semantic_search_tools=frozenset(
        {
            "mcp__sourcegraph__nls_search",
            "mcp__sourcegraph__deepsearch",
            "mcp__sourcegraph__deepsearch_read",
        }
    ),
)

OPENHANDS_REGISTRY = ToolRegistry(
    search_tools=frozenset(
        {
            "execute_bash",
        }
    ),
    edit_tools=frozenset(
        {
            "str_replace_editor",
        }
    ),
    code_nav_tools=frozenset(
        {
            "browser",
        }
    ),
    semantic_search_tools=frozenset(),
)

# ---------------------------------------------------------------------------
# Agent-based registry selection
# ---------------------------------------------------------------------------

_AGENT_REGISTRIES: dict[str, ToolRegistry] = {
    "openhands": OPENHANDS_REGISTRY,
}


def get_registry_for_agent(agent_name: str) -> ToolRegistry:
    """Return the appropriate :class:`ToolRegistry` for the given agent name.

    Performs case-insensitive substring matching against known agent names.
    Returns :data:`DEFAULT_REGISTRY` when no agent-specific registry is found.
    """
    agent_lower = agent_name.lower()
    for key, registry in _AGENT_REGISTRIES.items():
        if key in agent_lower:
            return registry
    return DEFAULT_REGISTRY
