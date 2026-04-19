# Agent Diagnostics

A behavioral taxonomy, annotation framework, and shareable dataset backend for analyzing why coding agents succeed or fail on benchmark tasks.

**11,995 trials. 4 models. 61 benchmarks. 40 failure categories across 11 dimensions.**

## What this does

Coding agents pass benchmarks for the wrong reasons and fail them for the wrong reasons. Pass/fail scores hide reward hacking, flawed tests, and lucky patches. This project extracts structured signals from agent trajectories, classifies failure modes, and provides a queryable dataset backend so you can actually understand what happened.

## Install

```bash
pip install agent-diagnostics
```

## Quick start: label your own traces

Point the tool at a directory of agent trial outputs. Each trial needs a `result.json` (and optionally a `trajectory.json`) in its own directory:

```
my-runs/
  trial-001/
    result.json          # must have: task_id or task_name, reward/score
    trajectory.json      # optional: list of agent steps with tool calls
  trial-002/
    result.json
    agent/
      trajectory.json    # also checks agent/ subdirectory
  ...
```

Then run the pipeline:

```bash
# Step 1: Extract signals from all trial directories into JSONL
agent-diagnostics ingest --runs-dir my-runs/ --output data/signals.jsonl

# Step 2: Classify failure modes (heuristic rules, instant)
agent-diagnostics annotate --signals data/signals.jsonl --output data/heuristic.json

# Step 3: LLM-assisted classification (reads actual trajectories)
agent-diagnostics llm-annotate --signals data/signals.jsonl --output data/llm.json \
    --sample-size 50 --model haiku --backend batch

# Step 4: Query results with SQL
agent-diagnostics query "SELECT model, count(*) as n,
  round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END)*100, 1) as pass_rate
  FROM signals GROUP BY model ORDER BY pass_rate DESC"
```

### What `result.json` looks like

The tool reads standard benchmark harness output. At minimum:

```json
{
  "task_name": "django__django-16527",
  "reward": 1.0,
  "agent_info": { "name": "claude-code" },
  "started_at": "2026-01-15T10:00:00Z",
  "finished_at": "2026-01-15T10:04:32Z"
}
```

Works out of the box with SWE-bench, OpenHands, and similar harnesses that write `result.json` per trial.

### What you get back

`signals.jsonl` — one row per trial with 31 structured fields:

```json
{
  "trial_id": "726c23ceb1ce7cf2...",
  "task_id": "django__django-16527",
  "model": "claude-sonnet-4-6",
  "reward": 1.0,
  "passed": true,
  "total_turns": 57,
  "tool_calls_total": 32,
  "search_tool_calls": 8,
  "edit_tool_calls": 12,
  "unique_files_read": 5,
  "unique_files_edited": 2,
  "duration_seconds": 272.0,
  "exception_crashed": false,
  "tool_call_sequence": ["read_file", "search", "edit_file", "..."],
  "..."
}
```

## Included dataset

The repo ships a Parquet export of 11,995 trials in `data/export/` (~1.5 MB):

| Model             | Trials | Pass rate |
| ----------------- | ------ | --------- |
| Claude Haiku 4.5  | 6,443  | 79.1%     |
| Claude Sonnet 4.6 | 4,564  | 73.2%     |
| Claude Opus 4.6   | 677    | 84.5%     |
| Claude Opus 4.5   | 253    | 71.9%     |

Query it immediately after cloning:

```bash
agent-diagnostics query "SELECT model, count(*) as trials FROM signals GROUP BY model"

# Or load directly
python3 -c "import pandas as pd; print(pd.read_parquet('data/export/signals.parquet').describe())"
```

### Pre-built queries

Five analysis queries are in `docs/queries/`:

```bash
agent-diagnostics query "$(cat docs/queries/per_model_outcomes.sql)"
agent-diagnostics query "$(cat docs/queries/benchmark_model_matrix.sql)"
agent-diagnostics query "$(cat docs/queries/annotation_cooccurrence.sql)"
agent-diagnostics query "$(cat docs/queries/tool_sequence_patterns.sql)"
agent-diagnostics query "$(cat docs/queries/eval_subset_export.sql)"
```

### Export your own Parquet

```bash
agent-diagnostics export --format parquet --out data/export/
```

Produces zstd-compressed Parquet with native `list<string>` columns, plus `MANIFEST.json` with schema version, row counts, SHA256 checksums, and source commit.

### Schema introspection

```bash
agent-diagnostics db schema --format markdown
agent-diagnostics db schema --format json
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

taxonomy = load_taxonomy()
names = valid_category_names()
```

## Annotation pipeline

### Heuristic annotation

23 rule-based classifiers that fire on signal patterns (e.g., `retrieval_failure` when search calls = 0 and files read = 0):

```bash
agent-diagnostics annotate --signals data/signals.jsonl --output heuristic.json
```

### LLM annotation

Reads actual trajectories and classifies with Claude. Supports `claude-code`, `api`, and `batch` (Message Batches API, 50% cheaper) backends:

```bash
agent-diagnostics llm-annotate --signals data/signals.jsonl --output llm.json \
    --sample-size 50 --model haiku --backend batch
```

### Ensemble (heuristic + classifier)

Two-tier: heuristic rules for structural categories, trained classifier for learned categories:

```bash
agent-diagnostics train --labels llm.json --signals data/signals.jsonl --output model.json
agent-diagnostics ensemble --signals data/signals.jsonl --model model.json --output ensemble.json
```

### Annotation store

All annotation writers can route through a shared `AnnotationStore` that enforces primary key uniqueness, atomic writes, and version consistency:

```bash
agent-diagnostics annotate --signals data/signals.jsonl --output heuristic.json \
    --annotations-out data/annotations.jsonl

agent-diagnostics ensemble --signals data/signals.jsonl --model model.json \
    --output ensemble.json --annotations-out data/annotations.jsonl
```

The store uses PK `(trial_id, category_name, annotator_type, annotator_identity, taxonomy_version)` so multiple annotators (heuristic, LLM, classifier, ensemble, human) can label the same trial without collision.

## Data formats

### signals.jsonl

One JSON object per line. 31 fields per trial including `trial_id` (stable SHA256-based), model, benchmark, reward, pass/fail, tool call counts/sequences, files read/edited, duration, error counts, and patch size.

### annotations.jsonl

Narrow-tall schema — one row per (trial, category, annotator):

| Column               | Description                                              |
| -------------------- | -------------------------------------------------------- |
| `trial_id`           | SHA256-based stable identifier                           |
| `category_name`      | Taxonomy category (e.g., `retrieval_failure`)            |
| `confidence`         | 0.0 to 1.0                                               |
| `evidence`           | Free-text explanation                                    |
| `annotator_type`     | `heuristic`, `llm`, `classifier`, `ensemble`, or `human` |
| `annotator_identity` | e.g., `heuristic:rule-engine`, `llm:haiku-4`             |
| `taxonomy_version`   | e.g., `3.0.0`                                            |
| `annotated_at`       | ISO 8601 timestamp                                       |

## CLI reference

```
agent-diagnostics ingest           Ingest trial directories into signals.jsonl
agent-diagnostics extract          Extract signals from a single trial directory
agent-diagnostics annotate         Heuristic annotation
agent-diagnostics llm-annotate     LLM-assisted annotation
agent-diagnostics train            Train per-category classifiers
agent-diagnostics predict          Predict with trained classifier
agent-diagnostics ensemble         Two-tier ensemble annotation
agent-diagnostics report           Generate Markdown + JSON report
agent-diagnostics calibrate        ECE, Brier, reliability diagrams vs a reference
agent-diagnostics validate         Validate annotations against schema
agent-diagnostics query            Run SQL against the dataset (DuckDB)
agent-diagnostics export           Export to Parquet with MANIFEST.json
agent-diagnostics manifest refresh Rewrite manifests.jsonl
agent-diagnostics db schema        Inspect table schemas
```

## Calibration metrics

Calibration asks a different question than agreement: not *do the annotators
pick the same categories?* but *when the annotator says a category is present
with confidence 0.8, is it present 80% of the time?*  Three proper scoring
rules are reported by `agent-diagnostics calibrate`:

- **Expected Calibration Error (ECE)** — `[0, 1]`, lower is better.  Predictions
  are bucketed into equal-width bins on `[0, 1]` (default 10).  For each bin,
  we compute the gap between the mean confidence and the observed accuracy;
  ECE is the sample-size-weighted average of those gaps.  An ECE of 0.0 means
  the confidences match empirical frequency exactly.  An ECE of 0.49 means
  the model's confidences are off by roughly 49 percentage points on average.
- **Brier score** — `[0, 1]`, lower is better.  Mean squared error between
  emitted confidence and the binary outcome.  Unlike ECE, Brier penalises
  individual-sample error (not just per-bin averages), so it disambiguates
  "overconfident on half, underconfident on the other half" from "well
  calibrated overall".
- **Reliability diagram** — per-bin counts, mean confidence, and observed
  accuracy.  Serialised as JSON for downstream plotting.  A perfectly
  calibrated annotator's diagram lies on the y = x diagonal.

The Markdown report also includes a **direction arrow** (`>>` overconfident,
`<<` underconfident, `=` well calibrated within 1pp).

### How to run

```bash
agent-diagnostics calibrate \
  --predictor data/heuristic.json \
  --golden-dir tests/fixtures/golden_corpus/ \
  --output-dir reports/calib/
```

`--predictor` is the annotation file whose emitted confidences are being
scored.  `--reference` takes a second annotation file as ground truth, or
`--golden-dir` composes the golden corpus's per-trial
`expected_annotations.json` files into the reference.  Outputs are
`calibration.md` and `calibration.json` in `--output-dir`.

Calibration is only meaningful on predictors that emit per-category
confidences.  Legacy `observatory-annotation-v1` files (produced before the
annotator started emitting `confidence`) default every present category to
`1.0` on read, which makes every assigned category look maximally
overconfident.  Run the pipeline with the current LLM annotator to produce
a `v2` file before interpreting the ECE / Brier output.

### Scope

These metrics *describe* how well calibrated the annotator's confidences are.
They do not fix miscalibration — post-hoc calibration (temperature scaling,
Platt, isotonic regression) is a separate concern and is not performed here.

## Contributing

We welcome contributions of agent trace data, new benchmark integrations, taxonomy refinements, and annotation tooling. If you're building evaluation infrastructure for coding agents, we'd love to talk.

## License

Apache-2.0
