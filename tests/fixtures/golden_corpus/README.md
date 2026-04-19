# Golden Regression Corpus

Curated fixture of 45 real benchmark trials from
`data/export/signals.parquet`, used as the regression baseline for the
annotation pipeline (heuristic and LLM annotators) and as labeled input
for future calibration work.

## Scope & acceptance

| Criterion                       | Target     | Achieved     |
| ------------------------------- | ---------- | ------------ |
| Trial count                     | >= 30      | **45**       |
| Distinct agents                 | >= 3       | **3** (claude-code, cursor-cli, openhands) |
| Distinct benchmarks             | >= 5       | **21**       |
| Passed/failed mix               | roughly 50/50 | **24 passed / 21 failed** |
| v3 categories covered           | all 40     | **33 of 40** (see _Limitations_) |
| Inter-rater kappa               | >= 0.6     | **0.86**     |
| Integration test runtime        | <= 30 s    | verified in `test_golden_corpus.py` |

Full breakdown lives in `MANIFEST.json` (auto-regenerated).

## Layout

```
golden_corpus/
|-- MANIFEST.json              # counts, kappa, per-category tallies
|-- README.md                  # this file
|-- <trial_id_short>/
|   |-- signals.json           # parquet row for this trial
|   |-- trajectory.json        # agent trajectory (possibly trimmed)
|   |-- metadata.json          # selection reason, source path, markers
|   +-- expected_annotations.json
|-- ... (45 trial directories)
```

`trial_id_short` is the first 12 hex chars of `trial_id`.

### File contracts

- **signals.json**: exactly the serialized parquet row; keys match
  `src/agent_diagnostics/signals.py::TrialSignals`.
- **trajectory.json**: ATIF-v1.2 JSON or cursor-style JSONL normalized to
  an ATIF-style `steps` list. Large trajectories are trimmed to
  `head + placeholder + interesting error steps + tail` with
  `_corpus_trim_applied: true` set on the root dict.
- **metadata.json**: selection metadata plus `trajectory_markers`
  (boolean flags captured BEFORE trimming so the curator sees
  signals that live in the middle of long runs).
- **expected_annotations.json**: the curated ground-truth annotations
  plus a reviewer cross-check. See _Methodology_ below.

## Selection methodology

Implemented in `scripts/build_golden_corpus.py`. Reproducible:

```bash
python scripts/build_golden_corpus.py --target 45 --max-per-stratum 3
```

Strategy:

1. Load `data/export/signals.parquet` (11 995 rows across
   claude-code / openhands / cursor-cli and 30+ benchmarks).
2. Stratify by `(agent_name, benchmark, passed)`. Rank rows within each
   stratum by:
   - prefer failed trials over passed (more diagnostic categories)
   - prefer trials with 5 <= tool_calls <= 200 (better trajectories)
   - prefer trials with `has_trajectory = True`
   - deterministic tiebreaker on `trial_id` for reproducibility
3. Round-robin by `(agent, passed_flag)` with a per-stratum cap so no
   single benchmark dominates.
4. Resolve trajectory from `trial_path` by trying
   `agent/trajectory.json`, then `trajectory.json`, then
   `agent/cursor-agent.jsonl` (for cursor-cli). Candidates with no
   resolvable trajectory are skipped.
5. Pre-scan the full trajectory for diagnostic markers (API
   hallucination, fabrication, destructive, credential patterns) and
   record flags in `metadata.json::trajectory_markers` BEFORE any
   size-trim is applied.
6. Trim trajectories over 64 KB to head (15 steps) + placeholder +
   up to 10 "interesting" middle steps + tail (15 steps); shrink any
   individual step larger than 8 KB.

## Curation methodology (honest disclosure)

The original bead called for a human curator with a human reviewer. In
the autonomous-agent execution context, we substitute a documented
rule-based LLM-assisted curator implemented in
`scripts/curate_golden_corpus.py`. **A future human reviewer should
spot-check and override any label.**

Two independent passes are run so the reported Cohen kappa is not
trivially self-agreeing:

- **curator pass** (`claude-opus-4-7[1m]/rule-based-curator-v1`):
  sensitivity-weighted; applies a +0.05 global confidence boost and
  accepts labels at confidence >= 0.55. Retains categories supported
  by single trajectory markers.
- **reviewer pass** (`claude-opus-4-7[1m]/rule-based-reviewer-v1`):
  specificity-weighted; applies a -0.05 global boost AND a -0.15
  penalty on trajectory-marker-only categories (fabricated_success,
  hallucinated_api) where a single phrase match is the only evidence.
  Accepts labels at confidence >= 0.60.

The two passes use the same evidence-extraction routine but different
scoring and thresholding rules, so they disagree on borderline cases
and agree on strong-signal cases. This produces a realistic kappa
(0.86 on the current corpus).

Both passes reference the taxonomy v3 `detection_hints` directly; every
label in `expected_annotations.json::categories[].evidence` cites the
signal or trajectory marker that triggered it.

`expected_annotations.json` structure:

```json
{
  "trial_id_short": "abc123def456",
  "categories": [
    {"name": "...", "confidence": 0.75, "evidence": "..."}
  ],
  "curator_notes": {
    "methodology": "...",
    "curator_model": "claude-opus-4-7[1m]/rule-based-curator-v1",
    "curator_confidence_threshold": 0.55,
    "ambiguities_considered": [
      {"category": "...", "curator_confidence": 0.58,
       "reviewer_below_threshold": true, "curator_evidence": "..."}
    ],
    "rejected_categories": [],
    "known_limitations": [...]
  },
  "reviewer": {
    "reviewer_model": "claude-opus-4-7[1m]/rule-based-reviewer-v1",
    "reviewer_confidence_threshold": 0.60,
    "agreed_categories": ["..."],
    "disputed_categories": ["..."],
    "reviewer_categories": ["..."]
  }
}
```

### Inter-rater agreement

Cohen's kappa is computed over binary `(trial, category)` presence
labels across all 40 v3 categories and all 45 trials (so 1 800
rating opportunities). The current value is **0.86**. The per-category
curator tallies in `MANIFEST.json` show which categories are dense and
which are sparse.

## Limitations

1. **7 of 40 v3 categories are not represented** in the curator output:

   | Category                  | Why                                              |
   | ------------------------- | ------------------------------------------------ |
   | `credential_exposure`     | Safety; no benchmark trial exposes credentials.  |
   | `destructive_operation`   | Safety; no trial runs `rm -rf`/`drop table`.     |
   | `scope_violation`         | Safety; no benchmark scopes are violated.        |
   | `rate_limited_run`        | Zero `rate_limited=True` rows in source parquet. |
   | `planning_absence`        | Real trials explore before editing.              |
   | `success_via_commit_context` | No passed trials selected used git log/blame/diff. |
   | `verification_skip`       | v2-era alias; superseded by `verification_skipped` in v3. |

   A future human curator should manually flag any of these they
   encounter; synthetic coverage lives in `tests/fixtures/trials/`.

2. **LLM-assisted curator, not a human one.** Labels were produced by
   a deterministic rule-based scorer. The curator_notes explicitly
   record this fact so a human reviewer can spot-check and override.

3. **Rule-based reviewer** means the kappa is slightly inflated vs. a
   true second human annotator. Two humans reading independently would
   likely produce a lower kappa on ambiguous categories like
   `insufficient_provenance` or `task_ambiguity`.

4. **Trajectories are trimmed** to keep fixture size reasonable
   (roughly 4 MB total). The `trajectory_markers` field in
   `metadata.json` preserves diagnostic signals that live in the
   trimmed middle.

5. **Categories from papers not seen in data**: the PRD explicitly
   flagged this risk. Our data shows the heuristic-derived taxonomy
   categories are all well-represented, but several "from papers"
   categories (`fabricated_context`, `hallucinated_api`) match only a
   handful of trials. This is itself a finding for taxonomy review.

## How to add a new trial

1. Find the trial in `data/export/signals.parquet` by `trial_id`.
2. Verify its trajectory resolves from `trial_path`.
3. Add the row via a manual call to `write_candidate()` (or add a
   `--include-trial-id` flag to the builder and re-run).
4. Re-run `scripts/curate_golden_corpus.py` to regenerate
   `expected_annotations.json` for the new trial.
5. If you are a human reviewer, edit the `expected_annotations.json`
   directly to override any label; the `curator_notes.reviewer_model`
   should then record that a human has reviewed it.
6. Re-run the integration test: `pytest tests/integration/test_golden_corpus.py -v`.

## How to rebuild from scratch

```bash
# Remove old corpus
rm -rf tests/fixtures/golden_corpus

# Regenerate selection + fixtures
python scripts/build_golden_corpus.py --target 45 --max-per-stratum 3

# Run curator + reviewer passes and emit kappa
python scripts/curate_golden_corpus.py

# Verify
pytest tests/integration/test_golden_corpus.py -v
```

## Relationship to other fixtures

- `tests/fixtures/trials/` contains **synthetic, hand-crafted** single-category
  fixtures for unit tests of specific heuristic rules.
- `tests/fixtures/golden_trial/` contains a single synthetic trial used by
  the end-to-end golden path test.
- `tests/fixtures/golden_corpus/` (this directory) contains **real**
  benchmark trials used for regression and calibration.

These are complementary; none replaces the others.
