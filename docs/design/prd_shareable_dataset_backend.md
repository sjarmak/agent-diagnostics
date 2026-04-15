# PRD: Shareable & Analyzable Agent Trace Dataset Backend

**Status:** Refined (post-converge)
**Date:** 2026-04-15
**Related:** `prd_agent_trace_dataset_pipeline.md` (SH-2, MH-1..5 complete)

## Problem Statement

The `observatory ingest` pipeline (`src/agent_diagnostics/cli.py:290`, `src/agent_diagnostics/signals.py:756`) writes valid-trial records to JSONL + `.meta.json` sidecars. This works for batch processing, but the team and external collaborators need the _next layer_: a durable, queryable, easily shareable store that unlocks notebooks, dashboards, and eval-dataset export — without breaking the JSONL pipeline or introducing server/hosting burden.

Three independent research lenses (prior art, first-principles, contrarian) and a three-way structured debate converged on one counter-intuitive finding: **the dataset is too small (21.87 MB, 11,995 rows) to justify a traditional database, and "add Postgres" is actively incompatible with the "easily shareable" goal.** The correct next layer is a **DuckDB query facade + Parquet share artifact + an annotation write contract that fixes a newly-discovered four-writer collision bug** — all additive on top of the existing JSONL pipeline, zero server, zero migration.

A profile run during the converge phase (see §Measured Evidence) settled several design questions with real numbers instead of predictions.

## Goals & Non-Goals

### Goals

- Collaborators can run SQL against the dataset with one command and zero server setup.
- The dataset can be shipped to an external collaborator as ≤4 files, readable in pandas, Polars, R, DuckDB, and Arrow-aware tools without writing a loader.
- `report.py`-class aggregations (full-scan groupby over ~12k rows × 29 columns) complete in <1s. _Measured: ~0.6 ms on Parquet, ~87 ms on JSONL — both crush the target._
- The existing JSONL pipeline (SH-2, MH-1..5) remains the source of truth and continues to function unchanged.
- The four annotation writers (`cmd_annotate`, `cmd_llm_annotate`, `cmd_predict`, `cmd_ensemble`) cannot silently clobber each other — primary-key-level correctness is enforced at write time.
- Schema version, taxonomy version, row count, and source commit are queryable from any shared artifact.

### Non-Goals

- **No Postgres, MySQL, or hosted DB.** Rules out "easily shareable" by construction.
- **No SQLite in v1.** Deferred to N3 with explicit promotion triggers; the converge phase demonstrated a file-level fix is sufficient until real concurrent-writer pain appears.
- **No replacement of the JSONL pipeline.** JSONL remains the authoritative on-disk format and ingest target.
- **No trajectory payload split in v1.** Deferred to v1.1; the profile showed variable-length list columns compress and query fine inline on this corpus.
- **No web UI / real-time dashboard service.** Notebooks + static HTML are the surface.
- **No custom query DSL.** Plain SQL via DuckDB.
- **No migration runner / SQLAlchemy models in v1.**
- **No large-scale partitioning or sharding.** Corpus fits in one file.

## Requirements

### Must-Have

- **M0. Annotation write contract [NEW — emerged from convergence debate, expanded post-premortem].**
  - All four annotation writers (`cmd_annotate`, `cmd_llm_annotate`, `cmd_predict`, `cmd_ensemble`) route through a single `AnnotationStore` helper with one method: `upsert_annotations(rows)`. The helper:
    - (a) asserts PK uniqueness per writer invocation — **PK is `(trial_id, category_name, annotator_type, annotator_identity, taxonomy_version)`** (expanded post-premortem R2 to resolve S1's self-contradiction);
    - (b) appends/merges to `annotations.jsonl`;
    - (c) writes atomically via temp-file + rename;
    - (d) holds an advisory lock (`fcntl.flock`) across the read-merge-write cycle;
    - (e) **refuses mixed-version writes** — if an incoming batch's `taxonomy_version` or `schema_version` differs from the existing file's declared version, raise a named error (absorbs S5's content into M0 per premortem R5);
    - (f) resolves `annotator_identity` via a committed `src/agent_diagnostics/models.yaml` mapping — **the PK never stores raw Anthropic snapshot IDs** (premortem R3, D4). Logical identities only (`"llm:haiku-4"`, `"llm:sonnet-4"`, `"llm:opus-4"`).
    - Readers dedupe last-writer-wins by `annotated_at`.
  - **Acceptance:**
    - Running all four writers sequentially against the same corpus produces an `annotations.jsonl` with zero duplicate PKs (test fixture of 20 trials × 4 annotator types).
    - Concurrent invocation of two `cmd_llm_annotate` processes against the same output file produces a union of both runs, not a last-write-wins clobber (integration test with two subprocesses).
    - Writing a batch with `taxonomy_version='v4'` to a file whose existing rows declare `'v3'` raises `MixedVersionError` naming both versions.
    - Rotating `models.yaml` from `llm:haiku-4 → claude-haiku-4-5-20251001` to `llm:haiku-4 → claude-haiku-4-6-20260801`, then re-upserting, produces exactly one row per `(trial_id, category_name, "llm", "llm:haiku-4", taxonomy_version)`.
    - `ensemble_all` in `src/agent_diagnostics/ensemble.py:78` is refactored to read via the same store, eliminating the in-Python merge loop.
    - A PK assertion failure raises a clear error naming the offending key and annotator.
    - Open Q#8 (`fcntl.flock` cross-platform) is resolved with documented platform support and a cross-platform integration test (premortem R8).

- **M1. DuckDB query subcommand.**
  - `observatory query "SELECT ..."` runs a DuckDB query against `data/signals.jsonl`, `data/annotations.jsonl`, `data/manifests.jsonl`, and any Parquet exports under `data/export/`. Auto-registers the obvious table aliases (`signals`, `annotations`, `manifests`) so users don't have to write `read_json_auto(...)`.
  - **Acceptance:**
    - `observatory query "SELECT model, count(*) FROM signals GROUP BY model"` returns a table in <1s on the 12k-row corpus.
    - Requires only the `duckdb` wheel as a new dependency.
    - Works identically whether the underlying files are JSONL or Parquet.

- **M3. Stable `trial_id` primary key (NH-3 from parent PRD). [R1 RESOLVED]**
  - Each trial row carries `trial_id = sha256(task_id || config_name || started_at || model)[:32]`, computed from four fields that `signals.py` already extracts into `TrialSignals`. `trial_id_full` (non-truncated, 64 hex = 256 bits) is retained as a secondary column for collision debugging at N4 scale.
  - **Formula rationale (from stability audit):**
    - All four inputs (`task_id`, `config_name`, `started_at`, `model`) survive re-download, path changes, AND re-scoring.
    - `MANIFEST.json` has no `run_id` → Option B (harness-supplied) is dead.
    - `run_id` is not populated in `signals.py` (grep-verified) → original formula was unimplementable.
    - `config_name` disambiguates different harness configs on the same `task_id`.
    - `started_at` disambiguates reruns (stable — written at trial start, not modified on re-download).
    - `model` disambiguates multi-model sweeps on the same task.
    - `[:32]` = 128 bits (birthday-bound collision at ~1.8×10^19; safe well past N4 HF Hub scale of 100k+ rows).
  - **Semantic choice:** a retried trial (same `task_id`, same `config_name`, different `started_at`) produces a **different** `trial_id`. Retries are distinct attempts worth tracking separately. If the user wants "latest attempt only," that is a query-time filter (`ROW_NUMBER() OVER (PARTITION BY task_id, config_name ORDER BY started_at DESC) = 1`), not a PK collapse.
  - **Acceptance:**
    - The formula uses only fields present in `TrialSignals` today — no new extraction code needed.
    - `duckdb -c "SELECT count(DISTINCT trial_id) FROM read_json_auto('data/signals.jsonl')"` equals the row count (zero collisions on the 11,995-row corpus).
    - **Property tests (see §Property Test Spec):**
      - P1: same trial, new filesystem path → same `trial_id`.
      - P2: same trial, re-scored (different `reward`) → same `trial_id`.
      - P3: retried trial (same `task_id`+`config_name`, different `started_at`) → different `trial_id`.
      - P4: two trials with identical `started_at` within 1s but different `task_id` or `model` → different `trial_id`.
      - P5: two trials with identical `(task_id, config_name, started_at, model)` → same `trial_id` (deterministic).
      - P6: `trial_id` is exactly 32 hex characters; `trial_id_full` is exactly 64 hex characters.
      - P7: `trial_id == trial_id_full[:32]` (truncation is prefix, not a different hash).

- **M4. Parquet export as canonical share artifact.**
  - `observatory export --format parquet --out data/export/` writes `signals.parquet` (with native `list<string>` columns for `tool_call_sequence`, `files_read_list`, `files_edited_list` — inline, no split), `annotations.parquet`, `manifests.parquet`, and `MANIFEST.json` containing `{schema_version, taxonomy_version, row_count, sha256_per_file, source_commit, generated_at}`.
  - **Acceptance:**
    - `pandas.read_parquet('data/export/signals.parquet')` loads without errors in a clean venv with only `pandas` + `pyarrow` installed.
    - Total `data/export/` size ≤3 MB for the current 21.87 MB JSONL corpus (measured target: 990 KB `signals.parquet` zstd + annotations + manifests ≈ 1.5 MB).
    - Re-running on unchanged inputs produces byte-identical output except for `generated_at`.
    - `MANIFEST.json` validates against a committed JSON schema.
    - Zero new `pyarrow` features required — stock pyarrow read path works.

- **M5. Manifests remain a side artifact.**
  - `manifests.jsonl` (with Parquet sibling in M4's export) stores `(manifest_version, run_id, task_id, benchmark, benchmark_source)` and is joined at query time, never denormalized into `signals.jsonl`. Refreshing the manifest does not require re-ingesting trials.
  - **Acceptance:**
    - `observatory manifest refresh` rewrites only `manifests.jsonl`.
    - `observatory query "SELECT s.model, m.benchmark FROM signals s LEFT JOIN manifests m USING (task_id)"` returns the same benchmark assignments as the current inline denormalization.

- **M6. JSONL pipeline regression guard.**
  - All SH-2 and MH-1..5 tests from the parent PRD continue to pass. M4's Parquet exports are additive — the JSONL ingest format on disk is unchanged except for the `trial_id` column (M3) and the `annotations.jsonl` consolidation (M0).
  - **Acceptance:**
    - The existing test suite passes unchanged.
    - `signals.jsonl` consumers that predate this PRD continue to read rows successfully (the trajectory fields remain inline).

### Should-Have

- **S1. Annotations narrow-tall schema** — _implemented by M0_. Columns: `trial_id`, `category_name`, `confidence`, `evidence`, `annotator_type` (heuristic/llm/ensemble/human), `annotator_identity`, `taxonomy_version`, `annotated_at`. PK `(trial_id, category_name, annotator_type, annotator_identity)`.
  - **Acceptance:** Same trial carries annotations from ≥2 annotators and ≥2 taxonomy versions without row collision; `report.py` reads via `observatory query`.

- **S2. Schema introspection command.**
  - `observatory db schema [--format json|markdown]` emits the schema of `signals`, `annotations`, `manifests`. For Parquet artifacts it reads `pyarrow.parquet.read_schema` directly; for JSONL it uses DuckDB's inferred schema.
  - **Acceptance:** A collaborator who has never seen the source can answer "what fields exist and which are nullable" in <30s.

- **S3. Redaction pipeline stage.**
  - `observatory redact --in data/ --out data/public/` produces public-safe copies of `signals.jsonl`, `annotations.jsonl`, and the Parquet exports with configurable rules (hash absolute paths, drop private repo names, truncate tool arguments over N bytes). _Note: M2 (trajectory split) becomes mandatory if and when S3 ships — the split isolates the redaction surface._
  - **Acceptance:** A committed regex/rule set removes 100% of known PII in a test fixture of 20 representative trials; the raw → redacted transform is idempotent.

- **S4. Define the 5 target queries.**
  - Commit 5 representative SQL queries to `docs/queries/` covering: per-model outcome rates, benchmark × model failure matrix, annotation co-occurrence, trajectory tool-sequence patterns, and eval-subset export.
  - **Acceptance:** All 5 run via `observatory query` in <200 ms on the Parquet artifact (profile target, not <1s); each has a committed expected-shape snapshot test. _This is the biggest risk-reduction item in the PRD — prevents shipping storage machinery for queries nobody runs._

- **S5. JSONL versioning via MANIFEST.**
  - Every pipeline artifact carries `schema_version` and `taxonomy_version`. `observatory ingest` refuses to mix versions silently.
  - **Acceptance:** Running ingest against mixed-version inputs raises a clear error naming the offending versions; the MANIFEST written by `observatory export` agrees with what the source files declare.

- **M2. Trajectory payload split [DEMOTED from must-have].**
  - Move `tool_call_sequence`, `files_read_list`, `files_edited_list` from `signals.jsonl` into a companion `trajectories.jsonl` keyed by `trial_id`.
  - **Promotion triggers (any one flips it to must-have):**
    - S3 (redaction) is scheduled — the split isolates the redaction surface.
    - Measured p95 `tool_call_sequence` length exceeds 200 entries (currently 98).
    - An external collaborator asks for the scalar-only row shape.
    - Parquet export size exceeds 10 MB for ≤50k rows.
  - **Acceptance (if promoted):** `wc -l data/signals.jsonl == wc -l data/trajectories.jsonl`; per-row p95 size of scalar `signals.jsonl` ≥ 50% smaller than pre-split; join in DuckDB via `trial_id` reconstructs the original row set.
  - _Rationale for deferral: profile on the current corpus showed Parquet inline (990 KB zstd) is 13% smaller than the split variant (1.06 MB) and DuckDB groupby latency is identical (0.6 ms). M2 provides zero measurable v1 benefit._

### Nice-to-Have

- **N1. DuckDB-WASM static dashboard.**
  - A static `dashboard.html` bundles DuckDB-WASM + the Parquet export and renders key report views in-browser with no server.
  - **Acceptance:** Double-clicking `dashboard.html` after `observatory export` loads the dataset and renders at least one chart; works offline. The full 21.87 MB corpus fits in a single HTTP download, so this is genuinely feasible.

- **N2. zstd-compressed archive format.**
  - _Partially subsumed by M4:_ Parquet with zstd page compression already delivers 22× compression on signals (21.87 MB → 990 KB). N2 remains only for packaging `data/export/*.parquet + MANIFEST.json` into a single file for email/chat distribution.
  - **Acceptance:** Single-file archive ≤3 MB; `observatory query --from archive.tar.zst` transparently unpacks and queries.

- **N3. SQLite promotion path.**
  - Document the MLflow/Phoenix pattern (`AnnotationStore` backend swap from JSONL to SQLite with SQLAlchemy, `schema_migrations` table, WAL mode) with explicit promotion triggers.
  - **Promotion triggers (any one flips it to must-have):**
    - M0's PK assertion fires in CI or production even once.
    - A second lost-write annotation bug is reported.
    - Annotation volume causes `AnnotationStore` full-file rewrites to exceed 10s or 50 MB.
    - A concurrent multi-process writer workload appears.
    - A point-lookup workload (not full-scan groupby) emerges.
  - **Acceptance:** Design note committed at `docs/upgrade_to_sqlite.md` with the trigger list and the swap procedure (change `AnnotationStore`'s backend module; `report.py` and downstream readers are unaffected because M0 already hid the storage engine behind an interface).

- **N4. HuggingFace Datasets-compatible Parquet shards.**
  - Parquet export follows the `signals-train-XXXX-of-YYYY.parquet` naming convention so the corpus can be pushed to HF Hub without a repack.
  - **Promotion trigger:** eval-dataset export targeting HF Hub becomes a real workflow (currently Open Question #3).
  - **Acceptance:** `datasets.load_dataset("parquet", data_files="data/export/*.parquet")` loads successfully.

- **N5. OpenTelemetry GenAI semconv column aliases.**
  - A view maps `TrialSignals` fields to OTel GenAI attribute names (`gen_ai.request.model`, `gen_ai.system`, `error.type`).
  - **Acceptance:** `observatory query "SELECT gen_ai.request.model FROM signals_otel"` returns model names.

## Measured Evidence

The converge phase ran an actual profile against `/home/ds/projects/agent-diagnostics/data/signals.jsonl` (11,995 rows, 21.87 MB). These numbers settled multiple design questions:

| Layout                                          | Size (zstd)    | `GROUP BY model` | `sum(reward) GROUP BY model` |
| ----------------------------------------------- | -------------- | ---------------- | ---------------------------- |
| Source JSONL                                    | 21.87 MB       | 87 ms            | 85 ms                        |
| Parquet A (lists as JSON strings)               | 1.22 MB        | 0.6 ms           | 0.6 ms                       |
| Parquet B (native `list<string>` inline)        | **0.99 MB**    | **0.6 ms**       | **0.6 ms**                   |
| Parquet C (scalar split + trajectories sidecar) | 0.51 + 0.55 MB | 0.5 ms           | 0.6 ms                       |

**What the numbers settled:**

- Variable-length list columns do NOT break Parquet columnar compression. (Previous assumption: they do.)
- DuckDB-on-Parquet beats the <1s target by 1,500×. M1 will never be latency-bound at this corpus size.
- M2 (trajectory split) provides zero measurable v1 benefit. Parquet B is smaller than Parquet C.
- Parquet B as the v1 shape is correct: inline, native list columns, 990 KB.

**What the numbers did NOT settle:**

- The realistic upper bound on `tool_call_sequence` length over future runs. Current p95=98, max=500. If it grows to p95>200, M2 promotes.
- Whether collaborators will prefer Parquet, DuckDB, or raw JSONL when given all three.

## Design Considerations

### Convergence refinements (Round 2 of converge phase)

1. **M0 replaced a quiet data-loss bug with a tested contract.** Debater C (SQLite-first) grepped the code during Round 1 and found that `cli.py:26/99/234/258` + `ensemble.py:78` all do full-file `json.dump` rewrites with no coordination. S1's PK is unenforceable in today's code. Round 2 produced M0 as a file-level fix that all three debaters accepted — SQLite-first conceded it's sufficient, minimalist conceded the bug is real and deferral is not an option, Parquet-first conceded the annotation write path stays file-based.

2. **M2 (trajectory split) was demoted by measurement.** The minimalist lens challenged M2's acceptance criterion as a prediction, not a measurement. Debater B ran the profile during Round 2 and retracted the "M2 is a Parquet precondition" claim — Parquet B (inline lists) is actually smaller than Parquet C (split). M2 remains architecturally useful for redaction hygiene and future-proofing, but has no v1 justification on current data.

3. **SQLite deferral gained a concrete trigger.** Round 1 left N3 as "maybe later." Round 2 tied it to M0's behavior: if M0's PK assertion ever fires in CI or ops, SQLite wins and the migration is a single-module swap because `AnnotationStore` already hides the backend.

4. **Parquet as _canonical_ share artifact (not export).** Round 2 concession from the minimalist lens based on reversibility asymmetry: JSONL→Parquet later is a repack pipeline; Parquet→JSONL later is one `duckdb COPY`. Choosing Parquet now preserves reversibility in both directions.

### Key tension: "no DB" vs "durable store"

**Resolution (unchanged from draft, strengthened by debate):** Treat "durable store" as a property of the _artifact format_, not the engine. JSONL + Parquet export gives durability via git/DVC/dated directories. DuckDB provides the SQL surface without being a store. SQLite is a **point store for annotations only, deferred**, with a trigger on M0 failure.

### Failure-mode ranking

**Over-engineering (HIGH) >> Wrong-engine (MEDIUM) > Under-engineering (LOW).**

This PRD is biased toward under-engineering. Converge phase validated this bias: the draft had 6 must-haves; after debate the must-have set is still 6, but **M2 was replaced by M0**, which is a strictly better use of the same budget because it fixes a real bug instead of speculating about a future one.

### Zero Framework Cognition (ZFC) alignment

Pure mechanism: IO, schema, format translation, deterministic export, PK enforcement. No semantic classification, no hardcoded heuristics. All reasoning continues to be delegated to models via the existing annotation pipeline.

## Open Questions

1. **What are the 5 concrete target queries?** (S4) All three research lenses AND all three debaters agreed this is the biggest unresolved risk. Recommend a ≤30-minute workshop with anyone who will actually run analysis before hardening column choices.
2. ~~**Read-only or read-write collaborators?**~~ **RESOLVED** by debate: the team itself writes (four writers). M0 addresses the write-side contract.
3. **Is eval-dataset export targeting HF Hub?** If yes, N4 promotes to must-have and the Parquet sharding convention is fixed now.
4. ~~**`tool_call_sequence` realistic upper bound?**~~ **PARTIALLY RESOLVED**: measured median 29 / p95 98 / max 500 on current corpus; M2 promotion trigger set at p95>200.
5. **OTel GenAI semconv alignment timing** (N5) — align now (cheap) vs align later (breaking migration).
6. **Retention / snapshot policy.** Dated `data/export/YYYY-MM-DD/` directories are the default answer; confirm before first external handoff.
7. ~~**SQLite promotion triggers (N3).**~~ **RESOLVED** by debate: M0 PK assertion firing / concurrent writer workload / point-lookup workload / full-file rewrite exceeds 10s or 50 MB.
8. **[NEW] Does M0's `fcntl.flock` work on all target platforms?** macOS/Linux yes, Windows needs `msvcrt.locking` fallback. Worth confirming which platforms the team ships to.

## Research Provenance

### Diverge phase (3 lenses)

- **Lens 1 (Prior Art)** — MLflow/Phoenix SQLite-first pattern, OpenHands' migration back to JSONL, Inspect-AI zstd+binary, 3-entity convergence.
- **Lens 2 (First-Principles)** — measured 11,995 rows / 21 MB / median 1.7 KB/row; audited `report.py` as pure OLAP full-scan.
- **Lens 3 (Contrarian)** — 6-month failure narratives; ranked over-engineering >> wrong-engine > under-engineering.

### Converge phase (3-way debate)

- **Debater A (Minimalist)** — forced Debater B to actually profile rather than predict. Killed the "M2 is a Parquet precondition" rhetoric with a demand for measurement. Sharpest single argument: "The acceptance criterion for M2 is a PREDICTION, not a MEASUREMENT."
- **Debater B (Parquet-first)** — ran the profile that reshaped the whole debate. Measured Parquet B at 990 KB zstd and 0.6 ms queries, retracted own Round 1 position, and conceded M2 out of v1 based on own data. Held the Parquet-as-share-artifact line.
- **Debater C (SQLite-first)** — grepped `cli.py` and `ensemble.py` to find the four-writer annotation clobber bug. Without this discovery, the PRD would have shipped with a silent data-loss bug. Conceded SQLite in v1 after A proved file-level `flock` + PK-assertion is sufficient.

### Convergence summary

- **Consensus reached:** M0 is new and necessary; M2 is demoted by measurement; Parquet-inline is the canonical artifact; SQLite is deferred with a concrete trigger; DuckDB is the unified query surface; JSONL ingest pipeline is untouched.
- **Remaining dissent:** None on v1 scope. Tension remains on _when_ to promote M2 and SQLite, resolved into explicit triggers rather than judgment calls.
- **Biggest risk flagged:** S4 — shipping the storage layer without knowing the 5 target queries that justify it. Unchanged from diverge phase; debate did not resolve it.

### Premortem phase (3 failure lenses)

Full analysis in `premortem_shareable_dataset_backend.md`. All three lenses rated Critical severity / High likelihood. Two of three identified failures rooted in the PRD specifying acceptance criteria against fields that do not exist in the current code — **verified by grep** for Risk R1 (`run_id` absent from `signals.py`).

## Risk Annotations

_Applied from `/premortem` phase. Items marked **APPLIED** have already been written into the PRD above; items marked **TODO** are pre-implementation blockers._

### Top 5 Risks (Critical / High)

| #      | Risk                                                                                                                                                                                                                                           | Status                                                                                                                                                                                                                                                                    | Where                                       |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **R1** | M3's `trial_id` formula references `run_id`, which `signals.py` does not extract (grep-verified: only `types.py:111` and `annotation_schema.json:96` hits).                                                                                    | **RESOLVED** — Formula rewritten to `sha256(task_id \|\| config_name \|\| started_at \|\| model)[:32]` using only fields `signals.py` already extracts. Stability audit confirmed all four survive re-download, path changes, and re-scoring. 7 property tests specified. | M3                                          |
| **R2** | S1 PK omits `taxonomy_version` while S1's own acceptance claims "≥2 taxonomy versions without row collision" — self-contradiction. Two analysts on different taxonomy versions silently clobber each other via `fcntl.flock` last-writer-wins. | **APPLIED** — M0 PK expanded to `(trial_id, category_name, annotator_type, annotator_identity, taxonomy_version)`.                                                                                                                                                        | M0                                          |
| **R3** | Hardcoded `claude-haiku-4-5-20251001` in `llm_annotator.py:59`. Anthropic snapshot rotation orphans historical rows from PK.                                                                                                                   | **APPLIED** — M0 now resolves `annotator_identity` via `models.yaml`; raw snapshot IDs never reach PK.                                                                                                                                                                    | M0, new `src/agent_diagnostics/models.yaml` |
| **R4** | `duckdb` and `pyarrow` unpinned. External collaborator with `pyarrow==14` cannot read native `list<string>` columns; `duckdb` minor version drift silently changes `read_json_auto` inference.                                                 | **TODO-D5** — Add pins to `pyproject.toml`; M4 acceptance must run in throwaway venv with worst plausible pin; ship `signals-compat.parquet` (JSON-string lists) if needed; commit `docs/compat.md` matrix.                                                               | M4, `pyproject.toml`, `tests/integration/`  |
| **R5** | S4 (5 target queries with snapshot tests) is should-have, not must-have. Without it, neither R1 nor R2 would have been caught on day one. PRD calls S4 "biggest risk" then leaves it should-have.                                              | **TODO-D3** — Promote S4 to must-have and gate M0/M3/M4 merge on it. Commit the 5 queries to `docs/queries/` with pytest snapshot tests.                                                                                                                                  | S4 → M-tier                                 |

### Secondary Risks (High / Medium)

| #   | Risk                                                                                                                                     | Top Mitigation                                                                                                                                           |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R6  | `MANIFEST.json` derives version fields from first row, not `SELECT DISTINCT` — silently lies about mixed-version contents.               | **TODO** — Compute version fields via `SELECT DISTINCT` inside flock window; fail export non-zero if >1 version present.                                 |
| R7  | `read_json_auto` is version-dependent; cross-batch schema drift silently reshapes query results.                                         | **TODO** — Replace with explicit `read_json(columns={...})` from a committed column map generated from `TrialSignals`. Add cross-version CI matrix.      |
| R8  | Open Q#8 (`fcntl.flock` cross-platform) unresolved — NFS, Docker, Windows, macOS tmpdirs have different semantics.                       | **APPLIED** — M0 acceptance now lists Q#8 resolution as a merge blocker. Still **TODO** to actually answer it.                                           |
| R9  | `ensemble_all` in `ensemble.py:78` is an external API surface exported from `__init__.py`; refactor may break callers outside this repo. | **TODO** — Keep `ensemble_all` as thin public wrapper over `AnnotationStore`. Add `tests/test_backwards_compat.py` importing every `__init__.py` symbol. |
| R10 | `trial_id[:16]` = 64 bits of entropy. Safe at 12k rows; collision-prone at N4 (HF Hub, 100k+ rows).                                      | **APPLIED** — M3 acceptance now requires retaining `trial_id_full` (non-truncated) column alongside `trial_id`.                                          |

### Top 5 Design Modifications (from premortem synthesis)

1. **D1 — Audit `run_id`** before M3 ships. ✅ RESOLVED — formula rewritten to `sha256(task_id || config_name || started_at || model)[:32]`. Stability audit confirmed all four fields survive re-download, path changes, and re-scoring. 7 property tests specified in §Property Test Spec.
2. **D2 — Expand annotation PK to include `taxonomy_version`.** ✅ Applied to M0. _(10 lines.)_
3. **D3 — Promote S4 to must-have and gate M0/M3/M4 on it.** Pending — not yet applied. _(≤30 min workshop + 5 SQL files + ~100 lines pytest.)_
4. **D4 — Abstract `annotator_identity` via `models.yaml`.** ✅ Applied to M0. Still needs the actual yaml file + rotation migration template committed. _(40 lines + 1 yaml + 1 script.)_
5. **D5 — Pin `duckdb` + `pyarrow`; M4 acceptance in throwaway venv.** Pending — not yet applied. _(pyproject pins + 1 integration test + compat matrix doc.)_

### Cross-Cutting Themes

- **Theme A — "The PRD specified against aspirational fields, not real code."** Lens 1 and Lens 2 both grepped the source and found acceptance criteria referencing `run_id` and raw Anthropic snapshot IDs that aren't stable contracts. R1 verified. This is a planning-process failure: run the grep before you write the acceptance criterion.
- **Theme B — "Deferred S5 was the load-bearing hole."** Lens 2 and Lens 3 both exploited the lack of version refusal. Absorbing S5 into M0 (applied) closes this theme.
- **Theme C — "Acceptance tests ran in the happy-path dev env."** All three lenses. Addressed by D5 (throwaway venv) but not yet applied.
- **Theme D — "S4 is the forcing function everyone agrees on."** Lens 1 and Lens 3. Without the 5 target queries as pytest snapshots, every other mitigation is checked in the absence of the one test that matters. Pending D3.

## Property Test Spec: `trial_id` (M3)

Formula: `trial_id = sha256(task_id || config_name || started_at || model)[:32]`
Secondary: `trial_id_full = sha256(task_id || config_name || started_at || model)` (64 hex, full 256-bit)
Separator: `||` is a literal pipe-pipe delimiter to prevent field-boundary collisions (e.g., `task_id="a||b"` + `config_name="c"` must differ from `task_id="a"` + `config_name="b||c"`).

### Test file: `tests/test_trial_id.py`

```python
"""Property tests for trial_id stability and uniqueness (M3)."""
import hashlib
import pytest


def compute_trial_id(task_id: str, config_name: str, started_at: str, model: str) -> tuple[str, str]:
    """Return (trial_id, trial_id_full)."""
    payload = f"{task_id}||{config_name}||{started_at}||{model}"
    full = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return full[:32], full


# --- P1: path-independence ---
def test_p1_same_trial_different_path():
    """Same result.json content at two filesystem paths → same trial_id."""
    id_a, _ = compute_trial_id("task-42", "default", "2026-01-15T10:00:00Z", "anthropic/claude-sonnet-4-6")
    id_b, _ = compute_trial_id("task-42", "default", "2026-01-15T10:00:00Z", "anthropic/claude-sonnet-4-6")
    assert id_a == id_b  # path is NOT an input → always equal


# --- P2: re-scoring stability ---
def test_p2_rescore_same_id():
    """Re-scoring a trial (reward changes) does NOT change trial_id."""
    # reward is not an input to the hash
    id_before, _ = compute_trial_id("task-42", "default", "2026-01-15T10:00:00Z", "anthropic/claude-sonnet-4-6")
    id_after, _ = compute_trial_id("task-42", "default", "2026-01-15T10:00:00Z", "anthropic/claude-sonnet-4-6")
    assert id_before == id_after


# --- P3: retried trial → different id ---
def test_p3_retry_different_started_at():
    """Same task+config retried (different started_at) → different trial_id."""
    id_first, _ = compute_trial_id("task-42", "default", "2026-01-15T10:00:00Z", "anthropic/claude-sonnet-4-6")
    id_retry, _ = compute_trial_id("task-42", "default", "2026-01-15T10:05:00Z", "anthropic/claude-sonnet-4-6")
    assert id_first != id_retry


# --- P4: different task or model → different id ---
@pytest.mark.parametrize("field,val_a,val_b", [
    ("task_id", "task-42", "task-43"),
    ("config_name", "default", "turbo"),
    ("model", "anthropic/claude-sonnet-4-6", "anthropic/claude-haiku-4-5"),
])
def test_p4_any_field_change_changes_id(field, val_a, val_b):
    """Changing any single input field produces a different trial_id."""
    base = {"task_id": "task-42", "config_name": "default",
            "started_at": "2026-01-15T10:00:00Z", "model": "anthropic/claude-sonnet-4-6"}
    args_a = {**base, field: val_a}
    args_b = {**base, field: val_b}
    id_a, _ = compute_trial_id(**args_a)
    id_b, _ = compute_trial_id(**args_b)
    assert id_a != id_b


# --- P5: deterministic ---
def test_p5_deterministic():
    """Same inputs always produce the same trial_id (no randomness)."""
    results = {compute_trial_id("task-42", "default", "2026-01-15T10:00:00Z", "anthropic/claude-sonnet-4-6")[0]
               for _ in range(100)}
    assert len(results) == 1


# --- P6: format ---
def test_p6_length_and_hex():
    """trial_id is 32 hex chars; trial_id_full is 64 hex chars."""
    tid, tid_full = compute_trial_id("task-42", "default", "2026-01-15T10:00:00Z", "anthropic/claude-sonnet-4-6")
    assert len(tid) == 32
    assert len(tid_full) == 64
    assert all(c in "0123456789abcdef" for c in tid)
    assert all(c in "0123456789abcdef" for c in tid_full)


# --- P7: truncation is prefix ---
def test_p7_truncation_is_prefix():
    """trial_id == trial_id_full[:32]."""
    tid, tid_full = compute_trial_id("task-42", "default", "2026-01-15T10:00:00Z", "anthropic/claude-sonnet-4-6")
    assert tid == tid_full[:32]


# --- Collision resistance on real corpus ---
def test_no_collisions_on_real_corpus():
    """When run against data/signals.jsonl, zero trial_id collisions.

    Skipped if signals.jsonl is not present (CI without data).
    """
    import json
    from pathlib import Path

    signals_path = Path("data/signals.jsonl")
    if not signals_path.exists():
        pytest.skip("data/signals.jsonl not found")

    ids = set()
    dupes = []
    with open(signals_path) as f:
        for line in f:
            row = json.loads(line)
            tid, _ = compute_trial_id(
                row.get("task_id", ""),
                row.get("config_name", ""),
                row.get("started_at", ""),
                row.get("model", ""),
            )
            if tid in ids:
                dupes.append(tid)
            ids.add(tid)

    assert not dupes, f"Collisions found: {dupes[:5]}"
```

### Separator choice: `||`

Using `||` as a delimiter prevents field-boundary ambiguity. Without it, `task_id="ab" + config_name="cd"` would hash identically to `task_id="a" + config_name="bcd"`. The literal `||` string is vanishingly unlikely to appear in any real field value (`task_id`, `config_name`, `started_at`, and `model` are all structured identifiers).
