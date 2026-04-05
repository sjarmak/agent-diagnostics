# Plan: unit-cli

## Steps

1. Create `src/agent_observatory/cli.py` — port from CSB with import changes and judge removal
2. Create `src/agent_observatory/__main__.py` — import from agent_observatory.cli
3. Update `pyproject.toml` — add `[project.scripts]` entry
4. Create `tests/test_cli.py` — verify importability, help output, no --judge flag
5. Run tests and fix failures
