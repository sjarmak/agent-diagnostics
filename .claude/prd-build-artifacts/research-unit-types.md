# Research: unit-types

## Codebase Conventions

- Python >=3.10, hatchling build
- Type annotations on all signatures
- Frozen dataclasses for immutable data (per project rules)
- `from __future__ import annotations` not used; uses `Optional`, `Union` from typing
- Tests use pytest with class-based organization (`TestXxx`)
- Imports: `from agent_observatory.taxonomy import ...`
- Line length: 99 (ruff config)

## Existing Schema Alignment

- `annotation_schema.json` defines: annotation (task_id, trial_path, reward, passed, categories, run_id, model, config_name, benchmark, annotator {type, identity}, notes, signals, annotated_at)
- `category_assignment`: name, confidence, evidence, calibration_status
- Annotator has `type` and `identity` fields

## Key Decisions

- TrialSignals: TypedDict with 26 keys as specified in acceptance criteria
- TrialInput: Protocol class defining what the observatory needs to consume
- CategoryAssignment: frozen dataclass with name, confidence, evidence (str|None)
- Annotation: frozen dataclass mirroring the JSON schema annotation object
- AnnotationDocument: frozen dataclass for the top-level document wrapper
- Use `tuple` for immutable sequences (categories in Annotation)
- Use Python 3.10+ union syntax (`str | None`) where possible
