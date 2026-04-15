Extract structured signals from a directory of agent trial outputs.

Ask the user for the path to their runs directory if not provided as $ARGUMENTS. Each trial should be in its own subdirectory with a `result.json` file. Optionally a `trajectory.json` (or `agent/trajectory.json`) for tool call analysis.

Steps:

1. Verify the directory exists and contains trial subdirectories with result.json files. Count them and tell the user what you found.
2. Run: `agent-diagnostics ingest --runs-dir <path> --output data/signals.jsonl`
3. Show a summary: number of trials ingested, pass/fail rate, models found.
4. Run a quick query to show the user what they got: `agent-diagnostics query "SELECT model, count(*) as trials, round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END)*100, 1) as pass_rate FROM signals GROUP BY model ORDER BY pass_rate DESC"`

If --manifest is available (the user has a MANIFEST.json for benchmark resolution), include it with `--manifest <path>`.
