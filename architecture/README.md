# Architecture diagram (LikeC4)

Architecture-as-code model of **Agent Diagnostics** (the Agent Reliability
Observatory), rendered with [LikeC4](https://likec4.dev). The model is the
source of truth across [`spec.c4`](spec.c4) (element kinds, tags, deployment
node kinds), [`model.c4`](model.c4) (the system), and [`views.c4`](views.c4)
(structure, walkthrough, and risk views), with the deployment model in
[`deployment.c4`](deployment.c4). The narrative companion is the repo-root
[`README.md`](../README.md).

Every element `link`s to its source (`src/agent_diagnostics/…`, `scripts/…`,
`data/…`) so any box in the explorer is one click from the code behind it.

## Delivery state is tagged, not guessed

Every element carries a tag so **evolving / planned / research work renders
distinctly from what is already built and exercised** (legend in `spec.c4`):

| Tag | Meaning | Render |
|---|---|---|
| `#built` | code path exists and is exercised by tests / the shipped run | solid |
| `#evolving` | built, but the contract/science is still moving | **amber** |
| `#planned` | designed; not yet implemented (or v1 is a stub) | dashed, dimmed |
| `#research` | speculative research track | dashed, indigo |

The package is built and tested end-to-end (v0.8.1, alpha; ruff + pytest in CI).
What is flagged `#evolving` is the *science / contract surface*, not missing
code: the **taxonomy** (v3 active, v1/v2 retained), the **label blender** trust
policy, **calibration** semantics, and the **golden-corpus curation** passes.
No `#planned` / `#research` elements are present — the model reflects shipped
reality rather than a roadmap.

## Views

**Structure** — the static map:

| View | Scope |
|---|---|
| `index` | system landscape — Agent Diagnostics in context of the harnesses, Anthropic, GitHub |
| `observatorySystem` | the system decomposed into containers (CLI, core package, scripts, data corpora) |
| `coreContainer` | the core package internals — extract / annotate / train / evaluate / serve |
| `llmView` | the LLM annotator: three backends (claude-code / api / batch), prompt + redaction, cache |
| `signalsView` | signal extraction internals — tool registry + content-hash cache |
| `scriptsView` | the golden-corpus + schema-docs scripts |
| `evolving` | the contracts still moving, with their dependents dimmed |
| `deployment` | where each piece runs — one local Python process over flat files, off-host LLM calls |

**Walkthrough flows** (dynamic / numbered-step views) — the narrative spine for
a design-review walkthrough:

| View | Flow |
|---|---|
| `ingestFlow` | raw trial dirs → 31-field signals (read → bucket tools → cache → write) |
| `labelFlow` | the label pipeline (heuristic + LLM → blend → train → ensemble) |
| `llmAnnotateFlow` | LLM annotation of one trial (leak-guard redaction → cache → backend → store) |
| `serveFlow` | query the shipped Parquet / generate a report / export |

**Risk lens:**

| View | Scope |
|---|---|
| `risks` | the `#risk`-flagged elements with each open question stated in-box |

`#risk` items surfaced in the model:

- **prompt + redaction** (`llm_annotator/prompt.py`) — leak prevention is a fixed
  redaction set; a new outcome-bearing signal field silently leaks the label
  unless added to `REDACTED_SIGNAL_FIELDS`.
- **calibration metrics** (`calibrate.py`) — legacy v1 annotation files have no
  per-category confidence and default to `1.0` on read, making every category
  look maximally overconfident; only meaningful on v2+ predictors.
- **annotation store** (`annotation_store.py`) — concurrent-writer safety relies
  on `fcntl.flock`, which is Unix-only; on Windows multi-writer use is unguarded.
- **shipped dataset** (`data/export`) — only `signals.parquet` ships; the
  MANIFEST records `annotations=0` / `manifests=0`, so the annotation/manifest
  tables are empty on a fresh clone until the pipeline is run.

### Running the walkthrough

For a design review, present in this order: `index` → `observatorySystem`
(orient on structure) → the four walkthrough flows in sequence (what actually
happens) → `deployment` (where it runs) → `risks` (what to probe) → `evolving`
(what's still moving). In `npx likec4 start`, the dynamic views animate
step-by-step and each view's notes panel carries the gotchas (the leak-guard
redaction set, the F1/ECE trust gate, the Parquet-preferred query path).

## Viewing & regenerating

```bash
# Interactive, hot-reloading explorer (recommended)
npx likec4 start architecture

# Re-export static PNGs (needs a one-time browser download:
#   npx playwright install chromium-headless-shell)
npx likec4 export png architecture -o architecture/exports

# Validate the model (strict — the source of truth for correctness)
npx likec4 validate architecture
```

### Viewing the interactive explorer over SSH (headless remote)

`likec4 start` serves a Vite dev server on `localhost:5173`. From a headless
remote, forward that port to your laptop and open it locally — three options,
easiest first:

1. **VS Code / Cursor Remote-SSH** — run `npx likec4 start architecture` in the
   integrated terminal; the editor auto-forwards 5173 and offers "Open in
   Browser". Nothing else to configure.
2. **SSH local port-forward** — on your laptop:
   ```bash
   ssh -N -L 5173:localhost:5173 user@remote   # leave running
   ```
   then on the remote `npx likec4 start architecture` and open
   <http://localhost:5173> locally. (Already in an SSH session? Add the tunnel
   without reconnecting: press `~C` then type `-L 5173:localhost:5173`.)
3. **Bind + reach directly** — `npx likec4 start architecture --listen 0.0.0.0`
   and browse to `http://<remote-ip>:5173` (only if that port is reachable /
   firewall-open; the tunnel in option 2 is safer).
