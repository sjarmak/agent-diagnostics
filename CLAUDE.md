# Agent Diagnostics

Behavioral taxonomy and annotation framework for analyzing why coding agents succeed or fail.

## Quick reference

```bash
agent-diagnostics ingest --runs-dir <path> --output data/signals.jsonl
agent-diagnostics annotate --signals data/signals.jsonl --output data/heuristic.json
agent-diagnostics query "SELECT model, count(*) FROM signals GROUP BY model"
agent-diagnostics export --format parquet --out data/export/
```

## Slash commands

- `/ingest <path>` — Point at a directory of agent traces, extract signals into JSONL
- `/label [signals_path]` — Run heuristic + optional LLM annotation pipeline
- `/explore <question>` — Ask a question about the dataset in natural language, get SQL results
- `/export` — Export to shareable Parquet format
- `/report` — Generate a reliability report with key findings

## Project structure

- `src/agent_diagnostics/` — main package (14 modules, 7k lines)
- `tests/` — 961 tests, 80%+ coverage
- `data/export/` — shipped `signals.parquet` (11,995 trials, 1.5 MB) + `MANIFEST.json`; `annotations.parquet` / `manifests.parquet` only appear when those JSONLs are populated
- `docs/queries/` — 5 pre-built SQL analysis queries
- `docs/design/` — PRD and premortem documents

## Key conventions

- Python 3.10+, ruff for linting, pytest for tests
- All data flows through JSONL (one JSON object per line)
- `trial_id = sha256(task_id||config_name||started_at||model)[:32]` is the stable join key
- Annotations use narrow-tall schema with PK `(trial_id, category_name, annotator_type, annotator_identity, taxonomy_version)`
- Taxonomy v3: 40 categories across 11 dimensions (see `src/agent_diagnostics/taxonomy_v3.yaml`)

## Test

```bash
python3 -m pytest tests/ -q
```
