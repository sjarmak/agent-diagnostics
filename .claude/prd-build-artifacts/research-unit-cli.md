# Research: unit-cli

## CSB Source Analysis

- `/home/ds/CodeScaleBench/observatory/cli.py` — 402 lines, 8 subcommands: extract, annotate, report, llm-annotate, train, predict, ensemble, validate
- `/home/ds/CodeScaleBench/observatory/__main__.py` — 5 lines, imports from `observatory.cli`

## Changes Required

1. All `from observatory.X` imports -> `from agent_observatory.X`
2. Remove `--judge` flag from llm-annotate subcommand
3. Remove `judge_trial` import, `run_judge` variable, judge scoring block, `judge_count` tracking from `cmd_llm_annotate`
4. Parser description: "Agent Reliability Observatory" (remove CodeScaleBench reference)
5. `__main__.py`: import from `agent_observatory.cli`
6. pyproject.toml: add `[project.scripts]` section

## Existing Modules Confirmed

All imported modules exist in `src/agent_observatory/`: signals, annotator, report, llm_annotator, classifier, ensemble, taxonomy
