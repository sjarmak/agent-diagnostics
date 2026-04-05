# Agent Diagnostics

A behavioral taxonomy and annotation framework for analyzing why coding agents succeed or fail on benchmark tasks.

## Install

```bash
pip install agent-diagnostics
```

Optional extras:

```bash
pip install agent-diagnostics[llm]        # LLM annotation (anthropic SDK)
pip install agent-diagnostics[validation] # JSON schema validation
pip install agent-diagnostics[dev]        # pytest, ruff, coverage
```

## Quick Start

### 1. Annotate a trial from signals

```python
from agent_diagnostics import annotate_trial, TrialSignals

# Build a signal dict (e.g. from your agent's logs)
signals: TrialSignals = {
    "task_id": "django__django-16527",
    "reward": 0.0,
    "passed": False,
    "search_tool_calls": 0,
    "unique_files_read": 0,
    "tool_calls_total": 3,
    "exception_crashed": False,
    "rate_limited": False,
}

categories = annotate_trial(signals)
for c in categories:
    print(f"  {c.name} (confidence={c.confidence}): {c.evidence}")
# retrieval_failure (confidence=0.9): No search tool calls and no files read
```

### 2. Extract signals from a trial directory

```python
from agent_diagnostics import extract_signals

# Point at a directory containing result.json + trajectory.json
signals = extract_signals("path/to/trial_dir")
print(signals["reward"], signals["tool_calls_total"])
```

### 3. Generate a reliability report

```python
import json
from agent_diagnostics import generate_report

with open("annotations.json") as f:
    annotations = json.load(f)

md_path, json_path = generate_report(annotations, output_dir="reports/")
print(f"Report: {md_path}")
```

## CLI

The package ships a CLI with 8 subcommands:

```bash
# Extract signals from trial directories
observatory extract --runs-dir runs/_raw --output signals.json

# Heuristic annotation
observatory annotate --signals signals.json --output heuristic.json

# LLM annotation (sample of trials)
observatory llm-annotate --signals signals.json --output llm.json \
    --sample-size 50 --model haiku --backend claude-code

# Train a classifier from LLM labels
observatory train --labels llm.json --signals signals.json --output model.json

# Predict with trained classifier
observatory predict --model model.json --signals signals.json --output predictions.json

# Two-tier ensemble (heuristic + classifier)
observatory ensemble --signals signals.json --model model.json --output ensemble.json

# Generate Markdown + JSON report
observatory report --annotations ensemble.json --output reports/

# Validate annotations against schema + taxonomy
observatory validate --annotations ensemble.json
```

Or via `python -m`:

```bash
python -m agent_diagnostics --help
```

## Taxonomy

23 categories across three polarities:

| Polarity | Count | Examples                                                                                |
| -------- | ----- | --------------------------------------------------------------------------------------- |
| failure  | 15    | `retrieval_failure`, `query_churn`, `decomposition_failure`, `edit_verify_loop_failure` |
| success  | 5     | `success_via_code_nav`, `success_via_semantic_search`, `success_via_decomposition`      |
| neutral  | 3     | `rate_limited_run`, `task_ambiguity`, `insufficient_provenance`                         |

```python
from agent_diagnostics import load_taxonomy, valid_category_names

taxonomy = load_taxonomy()  # v1 (flat) by default
names = valid_category_names()  # set of 23 category name strings
```

Both v1 (flat) and v2 (hierarchical by dimension) formats are bundled.

## Architecture

```
agent_diagnostics/
  taxonomy.py        # Taxonomy loader (v1/v2 YAML)
  types.py           # TrialSignals TypedDict, CategoryAssignment, Annotation
  tool_registry.py   # Injectable tool name registry (DEFAULT_REGISTRY)
  signals.py         # Extract signals from trial directories
  annotator.py       # 23-rule heuristic annotator
  classifier.py      # Pure-Python logistic regression (no numpy)
  ensemble.py        # Two-tier ensemble (heuristic + classifier)
  calibrate.py       # Agreement analysis, Cohen's kappa
  blend_labels.py    # LLM + heuristic label blending
  llm_annotator.py   # LLM annotation (claude-code + API backends)
  report.py          # Markdown + JSON report generator
  cli.py             # CLI entrypoint
```

## Custom Tool Registry

By default, tool classification uses Claude Code + Sourcegraph MCP tool names. For other agent harnesses:

```python
from agent_diagnostics import annotate_trial, ToolRegistry

my_registry = ToolRegistry(
    search_tools=frozenset({"grep", "find", "rg"}),
    edit_tools=frozenset({"file_write", "patch"}),
    code_nav_tools=frozenset({"goto_def", "find_refs"}),
    semantic_search_tools=frozenset({"semantic_search"}),
)

categories = annotate_trial(signals, tool_registry=my_registry)
```

## Extending Signal Extraction

For benchmark-specific metadata (suite detection, model resolution), inject callables:

```python
from agent_diagnostics import extract_signals

signals = extract_signals(
    trial_dir,
    suite_mapping={"csb_swebench": "swebench_lite"},
    benchmark_resolver=lambda path: "my_benchmark",
    model_keywords={"haiku": "claude-haiku-4-5"},
)
```

## License

Apache-2.0
