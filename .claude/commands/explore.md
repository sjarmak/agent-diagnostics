Answer a question about the agent trace dataset using SQL.

The user's question is in $ARGUMENTS. Translate it into a DuckDB SQL query and run it against the dataset.

Available tables:

- `signals` — one row per trial. See [docs/schemas/signals.md](../../docs/schemas/signals.md) for the full column list and types.
- `annotations` — one row per (trial, category, annotator). See [docs/schemas/annotations.md](../../docs/schemas/annotations.md) for the full column list and types.

Steps:

1. Write the SQL query that answers the user's question
2. Run it: `agent-diagnostics query "<sql>" --data-dir data/` (append `--format json` when the output needs to be piped into another tool; other formats: `jsonl`, `csv`, or the default `table`)
3. Present the results clearly. If the output is large, summarize the key findings.
4. Suggest a follow-up query if relevant.

Examples of questions and corresponding SQL:

- "which model is best?" -> SELECT model, count(*) as trials, round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END)*100, 1) as pass_rate FROM signals GROUP BY model ORDER BY pass_rate DESC
- "what are the hardest benchmarks?" -> SELECT benchmark, count(*) as trials, round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END)*100, 1) as pass_rate FROM signals GROUP BY benchmark HAVING count(\*) >= 10 ORDER BY pass_rate ASC LIMIT 20
- "how long do failed trials take?" -> SELECT CASE WHEN passed THEN 'passed' ELSE 'failed' END as outcome, round(avg(duration_seconds), 0) as avg_secs, count(\*) as n FROM signals GROUP BY outcome
- "what tools do agents use most?" -> SELECT tool_name, count(\*) as uses FROM (SELECT unnest(tool_call_sequence) as tool_name FROM signals) GROUP BY tool_name ORDER BY uses DESC LIMIT 20
