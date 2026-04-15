# Agent Diagnostics

A behavioral taxonomy, annotation framework, and shareable dataset backend for analyzing why coding agents succeed or fail on benchmark tasks.

**11,995 trials. 4 models. 61 benchmarks. 40 failure categories across 11 dimensions.**

## What this does

Coding agents pass benchmarks for the wrong reasons and fail them for the wrong reasons. Pass/fail scores hide reward hacking, flawed tests, and lucky patches. This project extracts structured signals from agent trajectories, classifies failure modes, and provides a queryable dataset backend so you can actually understand what happened.

The pipeline:

```
Trial directories (result.json + trajectory.json)
  -> observatory ingest        (extract 31 structured signals per trial)
  -> observatory annotate      (heuristic failure classification)
  -> observatory llm-annotate  (LLM-assisted classification)
  -> observatory ensemble      (heuristic + classifier ensemble)
  -> observatory export        (Parquet + MANIFEST.json share artifact)
  -> observatory query         (SQL via DuckDB, zero server)
```

## Dataset

The current corpus covers 4 Claude models across 61 benchmark suites:

| Model             | Trials | Pass rate |
| ----------------- | ------ | --------- |
| Claude Haiku 4.5  | 6,443  | 79.1%     |
| Claude Sonnet 4.6 | 4,564  | 73.2%     |
| Claude Opus 4.6   | 677    | 84.5%     |
| Claude Opus 4.5   | 253    | 71.9%     |

Each trial carries 31 structured signals including tool call sequences, files read/edited, duration, error counts, patch size, and a stable content-addressed `trial_id`.

### Query the dataset

```bash
pip install agent-diagnostics[query]   # adds duckdb + pyarrow

# Pass rates by model
observatory query "SELECT model, count(*) as trials,
  round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END)*100, 1) as pass_rate
  FROM signals GROUP BY model ORDER BY pass_rate DESC"

# Failure analysis
observatory query "SELECT model, count(*) FROM signals
  WHERE passed = false GROUP BY model"

# Tool usage patterns
observatory query "$(cat docs/queries/tool_sequence_patterns.sql)"
```

### Export as Parquet

```bash
# 21.87 MB JSONL -> ~1 MB zstd Parquet
observatory export --format parquet --out data/export/

# Readable in pandas, Polars, R, DuckDB, any Arrow tool
python3 -c "import pandas; df = pandas.read_parquet('data/export/signals.parquet'); print(df.shape)"
```

The export produces `signals.parquet`, `annotations.parquet`, `manifests.parquet`, and a `MANIFEST.json` with schema version, taxonomy version, row counts, SHA256 checksums, and source commit.

### Schema introspection

```bash
observatory db schema --format markdown   # human-readable
observatory db schema --format json       # machine-readable
```

## Install

```bash
pip install agent-diagnostics
```

Optional extras:

```bash
pip install agent-diagnostics[query]      # DuckDB + PyArrow (query & export)
pip install agent-diagnostics[llm]        # LLM annotation (Anthropic SDK)
pip install agent-diagnostics[validation] # JSON schema validation
pip install agent-diagnostics[dev]        # pytest, ruff, coverage
```

## Taxonomy

40 categories across 11 behavioral dimensions (v3):

| Dimension     | Categories | Examples                                                                           |
| ------------- | ---------- | ---------------------------------------------------------------------------------- |
| Retrieval     | 3          | `retrieval_failure`, `query_churn`, `context_window_overflow`                      |
| ToolUse       | 4          | `wrong_tool_selection`, `tool_argument_error`, `tool_misinterpretation`            |
| Reasoning     | 3          | `decomposition_failure`, `incorrect_root_cause`, `overconfident_diagnosis`         |
| Execution     | 5          | `edit_verify_loop_failure`, `syntax_error_loop`, `incomplete_implementation`       |
| Environment   | 4          | `exception_crash`, `rate_limited_run`, `environment_mismatch`                      |
| Faithfulness  | 2          | `task_misunderstanding`, `scope_drift`                                             |
| Metacognition | 5          | `premature_submission`, `excessive_exploration`, `sunk_cost_persistence`           |
| Integrity     | 2          | `test_file_modification`, `reward_hacking`                                         |
| Safety        | 3          | `data_exfiltration_attempt`, `sandbox_escape`, `destructive_operation`             |
| Strategy      | 6          | `success_via_code_nav`, `success_via_semantic_search`, `success_via_decomposition` |
| Observability | 3          | `insufficient_provenance`, `task_ambiguity`, `unreproducible_result`               |

```python
from agent_diagnostics import load_taxonomy, valid_category_names

taxonomy = load_taxonomy()   # loads latest version
names = valid_category_names()
```

## Annotation pipeline

### Heuristic annotation

23 rule-based classifiers that fire on signal patterns (e.g., `retrieval_failure` when search calls = 0 and files read = 0):

```bash
observatory annotate --signals data/signals.json --output heuristic.json
```

### LLM annotation

Reads actual trajectories and classifies with Claude. Supports `claude-code`, `api`, and `batch` (Message Batches API, 50% cheaper) backends:

```bash
observatory llm-annotate --signals data/signals.json --output llm.json \
    --sample-size 50 --model haiku --backend batch
```

### Ensemble (heuristic + classifier)

Two-tier: heuristic rules for structural categories, trained classifier for learned categories:

```bash
observatory train --labels llm.json --signals signals.json --output model.json
observatory ensemble --signals signals.json --model model.json --output ensemble.json
```

### Annotation store

All annotation writers can route through a shared `AnnotationStore` that enforces primary key uniqueness, atomic writes, and version consistency:

```bash
# Each writer appends to a shared narrow-tall JSONL file
observatory annotate --signals data/signals.json --output heuristic.json \
    --annotations-out data/annotations.jsonl
observatory ensemble --signals data/signals.json --model model.json \
    --output ensemble.json --annotations-out data/annotations.jsonl
```

The store uses PK `(trial_id, category_name, annotator_type, annotator_identity, taxonomy_version)` so multiple annotators (heuristic, LLM, classifier, ensemble, human) can label the same trial without collision.

## CLI reference

```
observatory extract         Extract signals from trial directories
observatory ingest          Filter -> extract -> enrich -> write JSONL pipeline
observatory annotate        Heuristic annotation
observatory llm-annotate    LLM-assisted annotation
observatory train           Train per-category classifiers
observatory predict         Predict with trained classifier
observatory ensemble        Two-tier ensemble annotation
observatory report          Generate Markdown + JSON report
observatory validate        Validate annotations against schema
observatory query           Run SQL against the dataset (DuckDB)
observatory export          Export to Parquet with MANIFEST.json
observatory manifest refresh  Rewrite manifests.jsonl
observatory db schema       Inspect table schemas
```

## Architecture

```
agent_diagnostics/
  signals.py           Signal extraction + trial_id computation (31 fields)
  types.py             TrialSignals TypedDict, CategoryAssignment, Annotation
  annotator.py         23-rule heuristic annotator
  classifier.py        Pure-Python logistic regression (no numpy)
  ensemble.py          Two-tier ensemble (heuristic + classifier)
  llm_annotator.py     LLM annotation (claude-code, API, batch backends)
  annotation_store.py  Narrow-tall JSONL store with PK enforcement + flock
  model_identity.py    Logical annotator identity resolution via models.yaml
  query.py             DuckDB query engine (JSONL + Parquet)
  export.py            Parquet export with MANIFEST.json
  report.py            Markdown + JSON report generator
  calibrate.py         Agreement analysis, Cohen's kappa
  blend_labels.py      LLM + heuristic label blending
  taxonomy.py          Taxonomy loader (v1/v2/v3 YAML)
  tool_registry.py     Injectable tool name registry
  cli.py               CLI entrypoint (14 subcommands)
```

## Data formats

### signals.jsonl

One JSON object per line. 31 fields per trial including `trial_id` (stable SHA256-based), model, benchmark, reward, pass/fail, tool call counts/sequences, files read/edited, duration, error counts, and patch size.

### annotations.jsonl

Narrow-tall schema: one row per (trial, category, annotator). Columns: `trial_id`, `category_name`, `confidence`, `evidence`, `annotator_type`, `annotator_identity`, `taxonomy_version`, `annotated_at`.

### Parquet export

`observatory export` produces zstd-compressed Parquet with native `list<string>` columns. The 21.87 MB JSONL corpus compresses to ~1 MB. Includes `MANIFEST.json` for provenance.

## Contributing

We welcome contributions of agent trace data, new benchmark integrations, taxonomy refinements, and annotation tooling. If you're building evaluation infrastructure for coding agents, we'd love to talk.

## License

Apache-2.0
