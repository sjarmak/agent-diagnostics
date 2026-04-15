# PRD: Agent Trace Dataset Pipeline

## Problem Statement

The agent-diagnostics observatory has 24,679 trial runs across multiple benchmarks, models, and agent frameworks — but no reliable pipeline to clean, validate, enrich, and analyze them. Current extraction is contaminated: harness summary files (12,657) outnumber actual trial results (12,022), inflating the dataset by ~2x with ghost rows. Reward defaults mask missing verifier data. The highest-prevalence failure category (retrieval_failure) is likely an artifact of trajectory-absent trials defaulting tool counts to zero. OpenHands trials (~20% of the dataset) have tool names invisible to the registry, silently zeroing out all tool-category signals. MANIFEST.json (mapping runs to benchmarks) exists but isn't wired in, leaving the benchmark field empty for all 24k trials.

The result: aggregate statistics are unreliable, comparative analytics are impossible, and the 500 LLM-labeled training examples may be contaminated with ghost data. Before any analysis or model training can be trusted, the extraction pipeline needs quality gates, the signals need proper denominators, and the enrichment infrastructure (which already exists at the library level) needs to be wired through the CLI.

## Goals & Non-Goals

### Goals

- Produce a clean, validated, enriched signals corpus from the 24,679 trial directories
- Eliminate ghost rows, distinguish missing data from zeros, and resolve benchmark metadata
- Enable model-vs-model, benchmark-vs-benchmark, and dimension-level comparative analysis
- Make the pipeline repeatable and incremental for new runs
- Support both trajectory-rich and trajectory-absent trials with proper denominator handling

### Non-Goals

- Building a web UI or dashboard (analysis outputs are files consumed by Python/pandas)
- Real-time streaming ingestion (batch pipeline is sufficient for periodic new runs)
- Replacing the LLM annotation system (the pipeline feeds it, doesn't replace it)
- Full statistical modeling (logistic regression, PCA) — those are analysis-layer concerns built on top of a clean pipeline, not part of the pipeline itself

## Requirements

### Must-Have

- **MH-1: Trial filter predicate**
  - Add `_is_valid_trial(data: dict) -> bool` to `signals.py` that requires `agent_info` key in result.json and rejects harness summaries
  - Exclude directories matching patterns: `__archived_invalid`, `__incomplete`, `__pre_sgenv_fix`, `__verifier_path_bug`, `__doubled_prefix`
  - Acceptance: `extract_all()` on the full \_raw/ directory produces ≤12,100 trials (not 24,679). Every output row has a non-null `task_id`.

- **MH-2: Nullable reward + provenance flags**
  - Change reward from `float` default-to-0.0 to `float | None` when no verifier result exists
  - Add `has_verifier_result: bool` field to TrialSignals
  - Acceptance: Trials without `verifier_result` in result.json have `reward=None` and `has_verifier_result=False`. Downstream `annotate_trial()` handles None reward gracefully (skips reward-dependent heuristics).

- **MH-3: Trajectory-aware denominators**
  - Add `has_trajectory: bool` (already exists) as a first-class grouping axis in reporting
  - Every category count in reports must show its proper denominator: full corpus for reward-only categories, trajectory corpus for trajectory-dependent categories
  - Add a `requires_trajectory: bool` property to each heuristic checker (or derive from taxonomy `signal_dependencies`)
  - Acceptance: `generate_report()` output shows separate "Trajectory-Dependent Categories" and "Reward-Dependent Categories" sections with correct denominators. `retrieval_failure` rate is computed against trajectory-available trials only.

- **MH-4: MANIFEST.json integration**
  - Parse MANIFEST.json to build `suite_mapping` and wire into `extract_signals()` via the existing `benchmark_resolver` parameter
  - For trials not in MANIFEST, derive benchmark from directory-name conventions (e.g., `csb_crossrepo_*` → `crossrepo`, `openhands_*` → `openhands`)
  - Acceptance: `benchmark` field is non-null for ≥90% of valid trials. The "benchmark field will be None" warning no longer fires for manifested trials.

- **MH-5: OpenHands tool registry**
  - Create `OPENHANDS_REGISTRY` mapping: `str_replace_editor` → edit_tools, `execute_bash` → shell_tools, `finish` → (ignored), `think` → (ignored)
  - Auto-select registry based on `agent_info.name` in result.json (claude-code → DEFAULT_REGISTRY, openhands → OPENHANDS_REGISTRY)
  - Acceptance: OpenHands trials have non-zero `edit_tool_calls` and `search_tool_calls` where applicable. `tool_call_sequence` entries are recognized by the active registry.

### Should-Have

- **SH-1: JSONL output format**
  - `extract_all()` and `annotate` write one JSON object per line (`.jsonl`)
  - Envelope metadata (schema_version, taxonomy_version, generated_at) written as a header comment or sidecar `.meta.json`
  - CLI commands accept both `.json` (legacy) and `.jsonl` (new) based on extension
  - Acceptance: `wc -l data/signals.jsonl` equals the number of trials. `head -1 data/signals.jsonl | python -m json.tool` produces a valid single-trial JSON object.

- **SH-2: `observatory ingest` command**
  - Single command that runs: filter → extract → enrich (from MANIFEST) → write JSONL
  - Accepts `--manifest`, `--runs-dir`, `--output`, `--state` flags
  - Incremental mode: tracks `{trial_path, mtime, size}` in state file, only re-extracts changed trials
  - Acceptance: Running `ingest` twice with no changes produces identical output and completes in <2s (mtime skip). Adding 10 new trial dirs and re-running processes only those 10.

- **SH-3: Category co-occurrence matrix**
  - Compute pairwise co-occurrence counts and phi coefficients for all assigned categories
  - Output as a symmetric matrix (JSON or CSV) suitable for heatmap visualization
  - Acceptance: Matrix is 40×40 (all v3 categories). Diagonal contains category prevalence. Off-diagonal phi coefficients are in [-1, 1].

- **SH-4: Dimension-level aggregation**
  - Roll up category assignments to parent dimension using taxonomy_v3.yaml structure
  - Output per-trial dimension scores and per-model/benchmark dimension failure rates
  - Acceptance: `generate_report()` includes a "Dimension Summary" section showing failure rate per dimension. JSON output includes `{model: {dimension: rate}}` structure.

### Nice-to-Have

- **NH-1: Content-hash cache for extraction**
  - Cache extracted signals keyed on `SHA256(result.json + trajectory.json contents)` to skip re-extraction of unchanged trials even without mtime tracking
  - Acceptance: Re-extracting the full corpus with warm cache completes in <5s.

- **NH-2: Pipeline DAG definition**
  - `pipeline.toml` declaring stages, inputs/outputs, and staleness rules
  - `observatory pipeline run` checks mtimes and runs only stale stages
  - Acceptance: `observatory pipeline run` with all stages up-to-date completes in <1s with "all stages up to date" message.

- **NH-3: Stable trial ID**
  - Derive `trial_id` from `SHA256(run_id + task_id + started_at)` as a portable primary key
  - Replace `trial_path` as join key in annotation files
  - Acceptance: Moving the data directory and re-running produces identical trial_ids.

## Build Sequence (from convergence debate)

Three positions debated (Data Quality First, Infrastructure First, Analysis-Driven) and converged on a phased approach:

### Phase 1: Surgical fixes + first analysis (parallel, days 1-2)

**Track A — Quality fixes:**

- MH-1: `_is_valid_trial()` predicate (require `agent_info` key, exclude known-bad dirs)
- MH-2: Nullable reward (`float | None`), `has_verifier_result` flag, update all heuristics for None-safety

**Track B — First analysis on clean partition:**

- Filter `has_trajectory=True` to get ~11k trials (implicitly excludes ghost rows)
- SH-3: Co-occurrence matrix on this subset
- SH-4: Dimension-level aggregation
- These analyses reveal which remaining fixes (MH-3/4/5) are load-bearing

**Rationale:** Analysis-advocate proved `has_trajectory=True` implicitly solves MH-1 for the analysis subset. Quality-advocate proved MH-2 is blocking even within that subset. Running both tracks in parallel maximizes time-to-first-insight while ensuring the data foundation is sound.

### Phase 2: Infrastructure + denominators (days 3-5)

- SH-2: `observatory ingest` command with MH-1 as first stage
- SH-1: JSONL output format
- MH-3: Trajectory-aware denominators in report.py
- Wire existing library APIs (`suite_mapping`, `benchmark_resolver`, `model_keywords`) through CLI

**Rationale:** Infra-advocate's key point — the ~2 hour cost of making MH-1 a composable CLI stage (vs. ad-hoc script) pays for itself on every subsequent filter. Quality-advocate's point about reproducibility was uncontested.

### Phase 3: Enrichment driven by Phase 1 findings (week 2)

- MH-4: MANIFEST integration — only if Phase 1 analysis reveals benchmark comparison is needed
- MH-5: OpenHands registry — only if OpenHands trials show up as analytical gaps
- Training label refresh — rerun Haiku labeling on the clean corpus (MH-1 must be stable first)

**Rationale:** Analysis-advocate's strongest point — "the real risk isn't dirty data, it's building the wrong clean." Defer enrichment until analysis reveals what's load-bearing. Quality-advocate correctly identified that training labels must not be refreshed until MH-1 is in place.

### Key debate resolution: contaminated training labels

Quality-advocate raised that the 500 Haiku-labeled examples were drawn from the full 24,679-row dataset (including ghost rows). Analysis-advocate correctly scoped this: the labels are contaminated for training purposes, but fresh analysis on the trajectory-available subset is unaffected. **Resolution:** Do not reuse existing training labels until MH-1 is applied. Fresh analysis in Phase 1 uses heuristic annotations only, not the trained classifier.

## Design Considerations

**Ghost row elimination vs. data loss:** The trial filter (MH-1) must be carefully validated — it should reject harness summaries but not accidentally filter out trials with unusual result.json schemas from newer harness versions. A dry-run mode that reports what would be filtered before actually filtering is essential.

**MANIFEST coverage gap:** MANIFEST covers only ~23% of trials by task_id. The fallback strategy (directory-name parsing) must be robust and auditable — every trial should get a `benchmark_source` field indicating whether it came from MANIFEST, directory convention, or was left unresolved.

**Nullable reward ripple effects:** Changing reward from `float` to `float | None` will break every heuristic that does `if reward == 0.0` or `if reward >= 1.0` without None checks. This is a wide-reaching change that needs systematic grep-and-fix across annotator.py, classifier.py, ensemble.py, and report.py.

**OpenHands registry completeness:** The tool-name mapping will evolve as OpenHands updates its tool definitions. The registry should be data-driven (loaded from a config file) rather than hardcoded, to accommodate new frameworks without code changes.

**Incremental vs. clean re-extract:** The first run should be a full clean extraction to establish baseline. Incremental mode is for subsequent runs. Don't optimize for incrementality before the quality gates are solid.

## Open Questions

1. Is MANIFEST.json intended as a curated "official" subset or is it simply stale? The answer determines whether it's the source of truth for benchmark assignment or just one input among several.
2. Are the 1,092 duplicate (task_name, trial_name) pairs legitimate re-runs or copy artifacts? Dedup strategy depends on this.
3. Were the 500 Haiku-labeled trials sampled representatively or selected on some criterion (e.g., only trajectory-available, only failures)? Sampling bias would undermine the blended training set.
4. Should `tool_call_sequence` (potentially hundreds of entries per trial) be stored inline in the signals JSONL or separated into a companion file to reduce payload size?

## Research Provenance

Three independent research agents explored this question from different lenses:

- **Data Engineering & Quality** — Found the critical ghost-row contamination (12,657 harness summaries mixed with 12,022 trials), OpenHands tool invisibility, and MANIFEST's 23% coverage. Highest operational urgency.
- **Analytical Design & Insight Generation** — Identified retrieval_failure inflation as a measurement artifact, proposed co-occurrence analysis and dimension-level aggregation. Highest strategic value.
- **Pipeline Architecture & Extensibility** — Discovered that enrichment APIs exist at the library level but the CLI never wires them. Proposed `observatory ingest` command and JSONL migration. Most actionable path.

**Key convergence:** All three agents independently flagged MANIFEST disconnection, trajectory-absent bias, and the need for quality gates before extraction. **Key divergence:** Whether to invest in statistical methods (analysis) or infrastructure (pipeline) first — resolved by sequencing infrastructure as prerequisite for trustworthy statistics.
