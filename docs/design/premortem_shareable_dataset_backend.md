# Premortem: Shareable Dataset Backend v1

**Date:** 2026-04-15
**Input:** `/home/ds/projects/agent-diagnostics/prd_shareable_dataset_backend.md` (post-converge)
**Lenses:** 3 (Technical Architecture, Integration & Dependency, Operational)

All three independent failure narratives rated **Critical severity / High likelihood**. Two of three identified failures rooted in the PRD specifying acceptance criteria against fields that **do not exist in the current code** — the planning process itself was ungrounded.

## 1. Risk Registry

| #   | Failure Lens             | Severity | Likelihood | Score | Root Cause                                                                                                                                                                                                                                                                                              | Top Mitigation                                                                                                                                                                                               |
| --- | ------------------------ | -------- | ---------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Technical Architecture   | Critical | High       | 12    | M3's `trial_id = sha256(run_id \|\| task_id \|\| started_at)` references `run_id`, which is NOT extracted by `signals.py` today (grep returns 0 hits). Implementation will improvise, producing instability.                                                                                            | Audit `run_id` existence before M3 ships; rewrite formula using fields that actually exist (`benchmark`, `task_id`, relative `trial_path`, `sha256(result.json)`). Add stability + collision property tests. |
| 2   | Operational              | Critical | High       | 12    | S1 PK `(trial_id, category_name, annotator_type, annotator_identity)` does NOT include `taxonomy_version`, but S1's own acceptance criterion claims "≥2 taxonomy versions without row collision." Two analysts on different taxonomy versions silently last-writer-wins each other under `fcntl.flock`. | Add `taxonomy_version` to the annotation PK (~10 lines). Make `AnnotationStore` refuse mixed-version writes — promotes S5 into M0's acceptance.                                                              |
| 3   | Integration & Dependency | Critical | High       | 12    | `claude-haiku-4-5-20251001` is hardcoded in `llm_annotator.py:59`. Anthropic rotates snapshot IDs; when it does, `annotator_identity` changes, every historical LLM row orphans from the new PK, and re-annotation double-writes or silently loses.                                                     | Abstract `annotator_identity` into a logical identity (`"llm:haiku-4"`) resolved from a committed `models.yaml`. Raw snapshot IDs never reach the PK.                                                        |
| 4   | Integration & Dependency | Critical | High       | 12    | `duckdb` and `pyarrow` are not pinned. `read_json_auto` schema inference is version-dependent; `pyarrow==14` cannot load native `list<string>` columns written by newer versions. External collaborators with older environments break.                                                                 | Pin `duckdb>=1.2,<1.3` and `pyarrow>=14,<18` in `pyproject.toml`. M4 acceptance must run in a throwaway venv with the "worst plausible collaborator pin." Ship a `signals-compat.parquet` variant if needed. |
| 5   | Operational + Technical  | High     | High       | 9     | S4 (5 target queries with snapshot tests) is should-have, not must-have. Without it, nothing catches either the trial_id collision (#1) or the taxonomy_version clobber (#2) until a collaborator publishes wrong numbers. The PRD flags it as "biggest risk" then leaves it should-have.               | Promote S4 to must-have and make it a merge blocker for M0/M3/M4. Snapshot tests on `count(DISTINCT trial_id)` and `count(*) WHERE taxonomy_version = 'v4'` would catch both #1 and #2 on day one.           |
| 6   | Operational              | High     | High       | 9     | `MANIFEST.json` derives `taxonomy_version` and `schema_version` from the first row it sees, not from `SELECT DISTINCT`. A mixed-version artifact silently lies about its own contents.                                                                                                                  | Compute version fields via `SELECT DISTINCT` inside the flock window; fail export non-zero if >1 version is present in any artifact.                                                                         |
| 7   | Integration & Dependency | High     | Medium     | 6     | `DuckDB read_json_auto` is unpinned — schema inference drifts across DuckDB versions, so the same JSONL file produces different column types on different team laptops.                                                                                                                                 | Replace `read_json_auto` with explicit `read_json(columns={...})` using a committed column map generated from `TrialSignals`. Add a cross-version CI matrix.                                                 |
| 8   | Operational              | High     | Medium     | 6     | Open Question #8 (`fcntl.flock` on all platforms) is still unresolved. Windows, NFS, Docker bind mounts, and macOS tmpdirs all have different `flock` semantics.                                                                                                                                        | Resolve Q#8 before shipping. Add an integration test running two subprocesses in different `taxonomy_version` configs on each supported OS.                                                                  |
| 9   | Integration & Dependency | Medium   | Medium     | 4     | Refactoring the 4 annotation writers may break `ensemble_all` in `ensemble.py:78`, which is exported from the package (external API surface, imported elsewhere).                                                                                                                                       | Keep `ensemble_all` as a thin public wrapper over `AnnotationStore.upsert_annotations`. Add backwards-compat test importing every symbol from `__init__.py`.                                                 |
| 10  | Technical Architecture   | Medium   | Medium     | 4     | `trial_id[:16]` = 64 bits of entropy. Safe on 12k rows today; risky at 100k+ (N4 HF Hub target). Birthday-bound collision becomes measurable.                                                                                                                                                           | Retain non-truncated `trial_id_full` column alongside `trial_id`. Cheap (22 bytes/row), makes collision debugging a SQL query.                                                                               |

## 2. Cross-Cutting Themes

### Theme A: "The PRD specifies against aspirational fields, not real code"

**Surfaced by:** Lens 1 (Technical Architecture), Lens 2 (Integration).
**Why this matters:** Two independent agents grepped the code and found that acceptance criteria reference fields that _don't exist in the current codebase_. Lens 1 found `run_id` is not extracted. Lens 2 found `annotator_identity` is de facto a raw Anthropic snapshot ID because that's what `llm_annotator.py:59` produces. The PRD was written as if these were already resolved. They aren't.
**Combined severity if exploited:** Critical. Every downstream contract (M0 PK, M3 stability, M4 Parquet schema) inherits this instability.

### Theme B: "Deferred S5 (version refusal) is the load-bearing hole"

**Surfaced by:** Lens 2, Lens 3.
**Why this matters:** The PRD explicitly demotes S5 (version refusal at write time) to should-have. Lens 3's entire operational failure hinges on this. Lens 2's model-ID rotation failure also leaks through because no write-time fence catches it. A 10-line version check inside the `AnnotationStore` flock window would prevent both failures.
**Combined severity if exploited:** Critical. Silent data divergence published to external collaborators.

### Theme C: "Acceptance tests run in the happy-path dev env"

**Surfaced by:** Lens 1, Lens 2, Lens 3.
**Why this matters:** All three agents independently noted that M4's acceptance criterion ("loads in a clean venv with `pandas` + `pyarrow` installed") is so generous that it validates the design against the author's own environment. Real failure modes live in version-pinned conda envs, mixed-version corpora, and concurrent runs from stale checkouts. The PRD doesn't test any of these.
**Combined severity if exploited:** High. Ships-green-then-fails-on-first-collaborator-handoff.

### Theme D: "S4 (5 target queries) is the forcing function everyone agrees on"

**Surfaced by:** Lens 1, Lens 3.
**Why this matters:** S4 is flagged in the PRD as "the biggest risk" but left as should-have. Both Lens 1 and Lens 3 independently point to snapshot tests on real queries as the mechanism that would catch their failure mode on day one. Promoting S4 to must-have and gating M0/M3/M4 on it is the single highest-leverage change.
**Combined severity if exploited:** High. Without S4, every other mitigation is checked in the absence of the one test that matters.

## 3. Mitigation Priority List

Ranked by (failures addressed × severity) / implementation cost.

| Rank | Mitigation                                                                                                                                                                | Addresses      | Cost                                                                     | Notes                                                      |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------- |
| 1    | **Grep `run_id` before M3 ships.** If not extracted, rewrite M3's formula using real fields (`benchmark`, `task_id`, relative `trial_path`, `content_hash(result.json)`). | #1, #10        | **Low** (30 min audit + formula change)                                  | Blocking.                                                  |
| 2    | **Add `taxonomy_version` to S1 PK.** Promote to M0 scope.                                                                                                                 | #2, #6         | **Low** (10 lines)                                                       | Fixes own self-contradiction in S1 acceptance.             |
| 3    | **Promote S5 (version refusal) into M0's acceptance.** `AnnotationStore.upsert_annotations` refuses mixed-version batches.                                                | #2, #3, #5     | **Low** (10 lines inside flock window)                                   | High-leverage single-file change.                          |
| 4    | **Promote S4 (5 target queries + snapshot tests) to must-have.** Hard v1 merge blocker.                                                                                   | #1, #2, #5, #7 | **Medium** (≤30 min workshop + 5 SQL files + pytest snapshots)           | The PRD's own "biggest risk" item.                         |
| 5    | **Abstract `annotator_identity` via `models.yaml`.** Logical identity, not raw snapshot ID. Commit a rotation migration script template.                                  | #3             | **Low-Medium** (40 lines + 1 yaml file + rotation script)                | Prevents quarterly blast radius on every Anthropic update. |
| 6    | **Pin `duckdb` and `pyarrow`; add external-env smoke test.** M4 acceptance runs in a fresh venv with the oldest supported `pyarrow`.                                      | #4, #7         | **Low-Medium** (pyproject pins + 1 integration test + compat matrix doc) | Converts "works on my laptop" into a contract.             |
| 7    | **Pin DuckDB schema explicitly.** Replace `read_json_auto` with `read_json(columns={...})` from a committed column map.                                                   | #1, #7         | **Medium** (~80 lines + generator + CI matrix)                           | Defends against silent DuckDB drift.                       |
| 8    | **Compute `MANIFEST.json` version fields via `SELECT DISTINCT`.** Fail export on mixed versions.                                                                          | #6             | **Low** (5 lines + 1 test)                                               | Belt-and-suspenders with mitigation #3.                    |
| 9    | **Resolve Open Q#8** (`fcntl.flock` cross-platform).                                                                                                                      | #8             | **Low-Medium** (1 integration test × N platforms)                        | Block ship until answered.                                 |
| 10   | **Retain `trial_id_full` alongside `trial_id[:16]`.**                                                                                                                     | #10            | **Low** (22 bytes/row)                                                   | Future-proofs for N4 (HF Hub scale).                       |
| 11   | **Keep `ensemble_all` as public wrapper; add backwards-compat import test.**                                                                                              | #9             | **Low** (1 test file)                                                    | Protects external API surface.                             |

## 4. Design Modification Recommendations (Top 5)

### D1. Audit `run_id` and rewrite M3's `trial_id` formula before anything else ships.

**What to change:** Search `src/agent_diagnostics/signals.py` for `run_id`. If absent, redefine M3's formula to use fields the extractor actually produces. Add property tests: (a) same trial, new path → same id; (b) retried trial → team picks "same" or "different" and encodes it; (c) two trials with colliding `started_at` → different ids.
**Addresses:** Risk #1, #10. Theme A.
**Effort:** 30 min audit + ~1 hour formula change + ~2 hours property tests.

### D2. Expand S1 PK to include `taxonomy_version` and collapse S5 into M0.

**What to change:** The annotation PK becomes `(trial_id, category_name, annotator_type, annotator_identity, taxonomy_version)`. `AnnotationStore.upsert_annotations` adds a 10-line check that refuses batches whose `taxonomy_version` differs from any existing row for the same PK. This resolves the self-contradiction in S1's own acceptance criterion and pulls S5's content into M0 at near-zero cost.
**Addresses:** Risk #2, #6. Theme B.
**Effort:** ~20 lines + 2 tests. Low.

### D3. Promote S4 (5 target queries with snapshot tests) to must-have and gate M0/M3/M4 merge on it.

**What to change:** Run the ≤30 min workshop. Commit 5 SQL files to `docs/queries/`. Add pytest snapshots that assert `count(DISTINCT trial_id)`, `count(*) GROUP BY taxonomy_version`, and the failure matrix row counts don't change silently. These become the regression surface that catches D1 and D2 failures on day one.
**Addresses:** Risk #1, #2, #5, #7. Theme D.
**Effort:** Low workshop + 5 SQL files + ~100 lines pytest.

### D4. Abstract `annotator_identity` through `models.yaml`; never store raw Anthropic snapshot IDs in the PK.

**What to change:** Create `src/agent_diagnostics/models.yaml` mapping logical names (`llm:haiku-4`, `llm:sonnet-4`, `llm:opus-4`) → current snapshot IDs. All four annotation writers resolve through this file. The PK stores only the logical name. Commit a rotation migration template so the next Anthropic deprecation is a one-line yaml bump, not a PK-breaking rewrite.
**Addresses:** Risk #3. Theme A (partially).
**Effort:** ~40 lines + 1 yaml + 1 migration script + 1 test. Low-medium.

### D5. M4 acceptance runs in a throwaway venv against the worst plausible collaborator pin; pin `duckdb` and `pyarrow`.

**What to change:** Add to `pyproject.toml`: `duckdb>=1.2,<1.3`, `pyarrow>=14,<18`. Add `tests/integration/test_external_collab.py` that `python -m venv`s a clean env, installs `pandas pyarrow==14.0.2`, and loads `signals.parquet`. Commit `docs/compat.md` with the tested matrix. If `pyarrow==14` can't read native `list<string>`, ship a `signals-compat.parquet` with JSON-string lists as a second artifact.
**Addresses:** Risk #4, #7. Theme C.
**Effort:** pyproject pins + ~30 lines integration test + doc. Low-medium.

## 5. Full Failure Narratives

### Lens 1 — Technical Architecture (Critical / High)

**What happened:**
In week two, the team shipped M3 (`trial_id = sha256(run_id || task_id || started_at)[:16]`) ahead of M0 because `AnnotationStore` needed a stable PK. The implementation had to paper over a gap: `run_id` is not a field the ingest pipeline extracts — it only appears as a nullable field in `src/agent_diagnostics/annotation_schema.json:96`. The first implementation fell back to `run_id = run_dir.name`, which changes every time a collaborator re-extracts the corpus. Meanwhile `started_at` turned out to be the harness's wall-clock, not a logical identifier — retries rewrote it. Within 10 days, collisions and instability compounded: two `mol-br7` trials produced the same `[:16]` hash because the `run_id=""` degenerate case collapsed entropy, and two retried trials second-resolution `started_at` collided after a clock-skew event. `AnnotationStore` dutifully raised its PK assertion (triggering N3's SQLite promotion), but `annotations.jsonl` had been rewritten 40+ times with drifting `trial_id` values.

The cascade hit `report.py` and M1 simultaneously. DuckDB's `read_json_auto` inferred `trial_id` as `VARCHAR` during the first batch, then silently dropped null rows from a subsequent batch's `GROUP BY` — the 5 target queries (S4) returned wrong counts for three weeks because S4's snapshot tests were never written (the PRD's own "biggest risk"). When the team ran `observatory export --format parquet`, pyarrow threw `ArrowInvalid: Unable to merge: Field trial_id has incompatible types`. The Parquet canonical artifact (M4) could no longer be regenerated. The October collaborator handoff was cancelled; the team spent the sprint writing a `trial_id` backfill — exactly the migration-runner work the PRD called a non-goal.

**Root cause:** M3 specified a `trial_id` formula referencing a `run_id` field that `src/agent_diagnostics/signals.py` never actually extracts, and `started_at` is not a logical key — so implementation had to improvise, and the PRD had no acceptance test that would have caught the improvisation.

**Warning signs:**

- `grep run_id src/agent_diagnostics/signals.py` returns zero hits.
- M3's acceptance has no test for "retried trial" or "colliding `started_at`" cases.
- `[:16]` hex = 64 bits. Safe at 12k rows, risky at N4 (HF Hub) scale.
- `read_json_auto` called with no explicit schema — cross-batch type drift is silent.
- S4 is flagged as "biggest risk" with no owner, no deadline, no snapshot tests.
- M0 enforces PK at the annotation layer but assumes upstream `trial_id` is stable.

---

### Lens 2 — Integration & Dependency (Critical / High)

**What happened:**
In mid-June, M4 shipped with `pyarrow==17.0.0` and `duckdb==1.2.2`, writing `signals.parquet` with native `list<string>` columns at the measured 990 KB zstd target. Three weeks later the first external collaborator — on a fresh `conda-forge` env with transitive pins from `snowflake-connector-python` — got `ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs` because their `pyarrow==14.0.2` pin couldn't read the format. The workaround (`pyarrow` upgrade) broke their notebook stack, so they asked for a JSONL dump — defeating M4's canonical-artifact goal. While triaging, Anthropic deprecated `claude-haiku-4-5-20251001` on 2026-08-15. CI's nightly `cmd_llm_annotate` 404'd; bumping the constant to `claude-haiku-4-6-20260801` meant every historical row had PK `(trial_id, ..., "llm", "claude-haiku-4-5-20251001")` that the new writer couldn't match.

The killing blow in September: someone ran `observatory query` on a machine `pip install --upgrade duckdb`'d to 1.4.0. DuckDB 1.4 changed `read_json_auto` inference of deeply nested union types, inferring `tool_call_sequence` entries with mixed-arity args as `STRUCT(...)` instead of `VARCHAR`. M1's auto-alias crashed on startup for upgraded users while 1.2.2 users saw a different schema for the same file. The batch-API `llm-annotate` path (commit 9d200a2) made it worse: two concurrent batch jobs held `fcntl.flock` individually, passed PK assertion individually, and produced a merged file DuckDB refused to ingest. By October the team reverted to the pre-M0 pipeline and disabled `observatory query` in the Makefile.

**Root cause:** The PRD pinned neither `duckdb` nor `pyarrow`, did not abstract the Anthropic model ID behind a stable annotator_identity, and treated external collaborator environments as out-of-scope for M4's acceptance criteria.

**Warning signs:**

- M4 acceptance tested in dev venv only; no collaborator-env simulation.
- `annotator_identity` is a free-form string with no model-rotation plan.
- `claude-haiku-4-5-20251001` is a bare literal in `llm_annotator.py:59` alongside `claude-sonnet-4-6` and `claude-opus-4-6` — two different ID schemes.
- DuckDB docs flag `read_json_auto` as "version-dependent"; PRD has no version contract.
- Converge profile ran on a single DuckDB build; no cross-version benchmark.
- Open Q#8 noted cross-env fragility but only for writers, not readers/export.

---

### Lens 3 — Operational (Critical / High)

**What happened:**
In late August, the LLM-annotator team kicked off a large rerun after bumping `taxonomy_version` from v3 to v4. The run straddled a shared laptop and "shared machine," with two analysts each starting a batch. `fcntl.flock` serialized writes correctly, so M0's PK assertion never fired. The problem: the two processes held _different_ `taxonomy_version` values in memory — analyst A on v4, analyst B on a stale checkout still pinned to v3. Both writes were valid. The PK `(trial_id, category_name, annotator_type, annotator_identity)` did not include `taxonomy_version`, so the later run's v3 rows last-writer-wins-clobbered v4 rows on `annotated_at` tiebreak whenever wall clocks disagreed. No error. No PK collision.

Two weeks later someone ran `observatory export --format parquet` and shipped `data/export/2026-09-02/` via Slack. The collaborator's notebook filtered on `taxonomy_version == 'v4'` and silently dropped ~40% of trials — because `MANIFEST.json` cheerfully reported `taxonomy_version: "v4"` (the export had read the first row and written that to the manifest). The collaborator published preliminary results in early October. The discrepancy was caught only when a second analyst tried to reproduce the numbers from a fresh ingest. By then the paper had been shared with two partner labs; the team spent three weeks reconstructing which rows came from which run, which they couldn't fully do because `annotated_at` had been overwritten during merge and analyst B's git history had been rebased.

**Root cause:** The PRD deferred S5 (version refusal) to should-have while making `AnnotationStore` the single write point across machines — so `fcntl.flock` prevented byte-level corruption but no write-time version check existed, and `taxonomy_version` was not part of S1's PK despite S1's own acceptance claim.

**Warning signs:**

- July dry run: row counts "a little different" between consecutive `observatory query` calls, shrugged off as caching.
- `cmd_llm_annotate` logs showed same `trial_id` annotated twice in 10 minutes; no alert because PK was structurally satisfied.
- `MANIFEST.json` version derived from a single row, not `SELECT DISTINCT`.
- A Beads ticket "should taxonomy_version be in the PK?" was triaged to S5 and forgotten.
- M4's byte-identical-on-rerun check gave false content-determinism confidence.
