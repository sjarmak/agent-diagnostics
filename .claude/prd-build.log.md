# PRD Build Log: Shareable Dataset Backend

## 2026-04-15T12:00:00Z — Decomposition complete

- 8 units across 4 layers
- Layer 0 (2 units): trial-id, models-yaml-and-deps
- Layer 1 (2 units): annotation-store, duckdb-query
- Layer 2 (2 units): writer-refactor, parquet-export
- Layer 3 (2 units): target-queries-and-schema, regression-guard
- Integration branch: prd-build/shareable-dataset-backend

## Layer 0 landed: trial-id (65f708f), models-yaml-and-deps (d2a9f39)

- trial_id = sha256(task_id||config_name||started_at||model)[:32] + trial_id_full (64 hex)
- compute_trial_id() in signals.py, TrialSignals TypedDict updated
- 10 property tests (P1-P7 + format, spec, collision)
- models.yaml with llm:haiku-4, llm:sonnet-4, llm:opus-4 mappings
- model_identity.py with resolve_identity() / resolve_snapshot()
- duckdb + pyarrow pinned in pyproject.toml query extra
- 848 tests passing

## Layer 1 landed: annotation-store (bcce2af), duckdb-query (36b8f02)

- AnnotationStore class: PK enforcement, atomic writes, fcntl.flock, MixedVersionError, identity resolution
- Narrow-tall JSONL schema: trial_id, category_name, confidence, evidence, annotator_type, annotator_identity, taxonomy_version, annotated_at
- query.py: run_query(sql, data_dir) with auto-registered signals/annotations/manifests
- observatory query subcommand in cli.py
- 892 tests passing

## Layer 2 landed: writer-refactor (ec83b81), parquet-export (b44815f)

- All 4 writers (annotate, llm-annotate, predict, ensemble) route through AnnotationStore via --annotations-out
- ensemble_all retains backwards-compat return format, public API preserved
- export.py: Parquet with zstd compression, native list<string> columns, deterministic output
- MANIFEST.json with schema/taxonomy versions, row counts, sha256 checksums
- observatory export + observatory manifest refresh subcommands
- 933 tests passing

## Layer 3 landed: target-queries-and-schema (350ecf2), regression-guard (1288134)

- 5 SQL queries in docs/queries/: per_model_outcomes, benchmark_model_matrix, annotation_cooccurrence, tool_sequence_patterns, eval_subset_export
- observatory db schema --format json|markdown
- Regression tests: trial_id/trial_id_full are only new fields, trajectory fields inline, annotation schema valid
- 961 tests passing

## PRD build complete

- **8/8 units landed**, 0 evictions, 1 pass
- **961 tests** (was 848 at start), all passing
- Integration branch: `prd-build/shareable-dataset-backend`
