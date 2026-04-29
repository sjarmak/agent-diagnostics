# Classifier trio slash commands — design note

**Status:** decided
**Author:** worker (gc-65752) for [agent-diagnostics-hnn](#)
**Parent epic:** agent-diagnostics-y84 — slash commands for remaining CLI subcommands
**Date:** 2026-04-29

## Context

The CLI exposes 16 subcommands; only 5 have slash-command wrappers today:
`/ingest → ingest`, `/label → annotate (+llm-annotate)`, `/explore → query`,
`/export → export`, `/report → report`. The classifier trio
(`train`, `predict`, `ensemble`) is missing wrappers and is the
highest-priority gap because it is the production labeling path once a
project has even a small batch of LLM labels.

The decision is whether to introduce three standalone commands
(`/train`, `/predict`, `/ensemble`) or fold the trio into `/label` via a
`--mode` flag (`{heuristic, llm, train, predict, ensemble}`).

Relevant facts:

- `/label` is **already** an orchestrator: it runs `annotate` (heuristic)
  unconditionally, then prompts the user for `llm-annotate`. The supposed
  1-to-1 mapping between slash commands and CLI subcommands is not
  actually preserved today. (`.claude/commands/label.md`)
- `train`, `predict`, `ensemble` have very different argparse signatures
  (`src/agent_diagnostics/cli.py:1140-1201`):
  - `train` requires `--labels` (LLM annotation JSON) and `--signals`,
    plus training hyperparameters (`--lr`, `--epochs`, `--min-positive`,
    `--eval`). Output: a model JSON.
  - `predict` requires `--model` and `--signals`. Output: annotation JSON
    (and optional `--annotations-out` for narrow-tall rows).
  - `ensemble` requires `--model` and `--signals`, with classifier
    gating (`--threshold`, `--min-f1`). Output: annotation JSON.
- The output **type** differs: `train` produces a *model artifact*;
  `predict` and `ensemble` produce *annotations* (the same artifact class
  as heuristic/LLM annotation). This is the key distinction the design
  needs to honor.
- Typical invocation pattern: `train` once (or rarely, when the LLM-label
  pool grows); then `predict` for evaluation or `ensemble` for production
  labeling, repeated each time signals are refreshed.

## Options considered

### Option A — three standalone slash commands

`/train`, `/predict`, `/ensemble` each map 1-to-1 to its CLI subcommand.

**Pros**
- Matches the precedent the bead description cites (5 existing wrappers,
  one per subcommand).
- Discoverable at the slash-command palette; intent is explicit.
- Each command can have its own focused interactive flow (e.g. `/train`
  asking for hyperparameters; `/ensemble` asking for thresholds).

**Cons**
- Adds three top-level commands and conflates lifecycle stages
  (model production vs annotation production) at the same nesting level.
- `/predict` is rarely the right call in production — `/ensemble`
  subsumes it. Listing both invites users to pick the wrong one.

### Option B — fold all into `/label --mode`

Single command, mode flag selects the annotator:
`{heuristic, llm, train, predict, ensemble}`.

**Pros**
- Single discovery point: "labeling lives at `/label`".
- Easy to teach: pick a mode, get annotations.

**Cons**
- `train` does **not** produce annotations — it produces a model JSON.
  Folding it into `/label` mislabels the workflow and breaks the user's
  mental model the moment they look at the output.
- `train`'s argument surface (LLM label file, hyperparameters, eval flag)
  has zero overlap with the other modes; an interactive `/label` would
  branch heavily on `--mode`, becoming a switch statement masquerading as
  a command.
- Hides a non-trivial lifecycle stage (model training) inside what users
  reach for to label trials.

### Option C — hybrid (RECOMMENDED)

- **`/train`** — standalone slash command. Train a classifier model from
  existing LLM labels. Distinct workflow, distinct output type
  (model JSON), distinct argument surface.
- **Extend `/label`** to cover `predict` and `ensemble` as additional
  annotator modes. Default behavior unchanged (heuristic, then prompt for
  LLM). When a trained model exists at `data/model.json`, the prompt
  surface adds: "run ensemble (heuristic + classifier)?" and exposes
  `predict` only as an explicit opt-in for sanity checks.

**Pros**
- Aligns command boundaries with **output types**: `/label` produces
  annotations; `/train` produces a model. Users learn one rule.
- Preserves the existing `/label` UX (heuristic-then-LLM) while adding
  the two ensemble paths users will actually want.
- Keeps the slash-command palette small and meaningful: 6 commands
  instead of 8, no `/predict` clutter when `/ensemble` is the
  production path.
- Argument surfaces stay focused: `/train` owns hyperparameters;
  `/label` modes share the `--signals` and annotations-output contract.

**Cons**
- `/label` grows in scope. Mitigation: keep the interactive prompt
  pattern that's already there — users see only the modes that apply
  given the artifacts present in `data/`.
- Slight asymmetry: the bead description cites a 1-to-1 mapping
  precedent, which Option C breaks for `predict`/`ensemble`. Counter:
  `/label` itself already breaks 1-to-1 today.

## Recommendation: Option C (hybrid)

**Add `/train` as a new standalone slash command. Extend `/label` to
also drive `predict` and `ensemble`.**

Rationale, in order of weight:

1. **Output-type alignment.** A slash command should map to a kind of
   thing the user gets back. `/label` returns annotations;
   `/ingest` returns signals; `/export` returns Parquet; `/report`
   returns markdown. Folding `train` into `/label` violates that
   invariant — `train` returns a model. `predict` and `ensemble`
   return annotations and therefore fit naturally under `/label`.
2. **Lifecycle separation.** Model training is a rare, deliberate step
   (you train once per LLM-label batch). Annotation is the frequent
   step. Surfacing them at the same level conflates cadence.
3. **Production path clarity.** `/ensemble` is the right call once a
   model exists; `/predict` is mostly diagnostic. Hiding `predict`
   inside `/label` (rather than promoting it to a top-level command)
   nudges users toward `ensemble` by default.
4. **Existing precedent.** `/label` already orchestrates two CLI
   subcommands. The 1-to-1 mapping the bead invokes is descriptive,
   not normative — and the file that breaks it is the very one we're
   extending.

## CLI argparse implications

No CLI argparse changes required. The three subparsers (`train`,
`predict`, `ensemble`) stay as they are — each has its own focused
flag set, which is correct. The slash command layer adapts to the
existing CLI rather than the other way round.

Specifically:

- `/train` invokes `agent-diagnostics train --labels ... --signals ...
  --output data/model.json` plus optional hyperparameters.
- `/label` (extended) invokes whichever of `annotate`, `llm-annotate`,
  `predict`, or `ensemble` the user picked, mapping interactive choices
  to subprocess flags. The argparse tree stays unchanged.

If we later want shared flags (e.g. `--data-dir`, `--annotations-out`)
across the annotation-producing subcommands, that's a separate
refactor and not blocked by this design.

## Follow-ups (not in scope here)

These are deliberately not decided in this note; they are the
implementation beads that should cite this design:

1. **Implement `/train`** — write `.claude/commands/train.md` with
   interactive prompts for `--labels`, `--signals`, hyperparameters,
   and `--eval`. Show training summary (n classifiers, skipped
   categories, train accuracy per category).
2. **Extend `/label`** — modify `.claude/commands/label.md` to detect
   whether a trained model exists at `data/model.json` (or the path
   the user supplies) and offer ensemble/predict as additional steps
   after the heuristic+LLM prompt block. Default to `ensemble` when
   prompted; offer `predict` only as a "sanity check" branch.
3. **Doc updates** — once both land, refresh the slash-command list in
   `CLAUDE.md` and the project README.
4. **Open question for follow-up:** should `/label` auto-suggest
   running `/train` when it detects ≥ N LLM labels and no model file?
   Defer until we see usage; do not pre-build the heuristic.

## Decision log

- Considered and rejected Option A (three standalone commands):
  promotes `/predict` to top-level visibility despite being mostly
  diagnostic; doubles classifier-related commands without buying
  clarity.
- Considered and rejected Option B (fold all into `/label --mode`):
  breaks output-type alignment by hiding model training inside the
  labeling command.
