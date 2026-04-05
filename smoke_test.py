"""Smoke test for agent-diagnostics PyPI package."""

from agent_diagnostics import (
    load_taxonomy,
    annotate_trial,
    TrialSignals,
    ToolRegistry,
    DEFAULT_REGISTRY,
    extract_signals,
    generate_report,
)

# Taxonomy loads with 23 categories
t = load_taxonomy()
print(f"{len(t['categories'])} categories loaded")

# Heuristic annotator works on a hand-built signal dict
signals: TrialSignals = {
    "reward": 0.0,
    "passed": False,
    "search_tool_calls": 0,
    "unique_files_read": 0,
    "exception_crashed": False,
    "rate_limited": False,
}
cats = annotate_trial(signals)
print(f"Annotated: {[c.name for c in cats]}")

# Tool registry
print(f"Default tools: {len(DEFAULT_REGISTRY.all_tools)} registered")

print("\nAll OK!")
