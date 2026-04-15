Answer a question about the agent trace dataset using SQL.

The user's question is in $ARGUMENTS. Translate it into a DuckDB SQL query and run it against the dataset.

Available tables:

- `signals` — one row per trial. Key columns: trial_id, task_id, model, benchmark, reward, passed, total_turns, tool_calls_total, search_tool_calls, edit_tool_calls, unique_files_read, unique_files_edited, duration_seconds, exception_crashed, rate_limited, tool_call_sequence (list), files_read_list (list), files_edited_list (list), patch_size_lines, trajectory_length, config_name, agent_name, benchmark_source
- `annotations` — one row per (trial, category, annotator). Columns: trial_id, category_name, confidence, evidence, annotator_type, annotator_identity, taxonomy_version, annotated_at

Steps:

1. Write the SQL query that answers the user's question
2. Run it: `agent-diagnostics query "<sql>" --data-dir data/`
3. Present the results clearly. If the output is large, summarize the key findings.
4. Suggest a follow-up query if relevant.

Examples of questions and corresponding SQL:

- "which model is best?" -> SELECT model, count(*) as trials, round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END)*100, 1) as pass_rate FROM signals GROUP BY model ORDER BY pass_rate DESC
- "what are the hardest benchmarks?" -> SELECT benchmark, count(*) as trials, round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END)*100, 1) as pass_rate FROM signals GROUP BY benchmark HAVING count(\*) >= 10 ORDER BY pass_rate ASC LIMIT 20
- "how long do failed trials take?" -> SELECT CASE WHEN passed THEN 'passed' ELSE 'failed' END as outcome, round(avg(duration_seconds), 0) as avg_secs, count(\*) as n FROM signals GROUP BY outcome
- "what tools do agents use most?" -> SELECT tool_name, count(\*) as uses FROM (SELECT unnest(tool_call_sequence) as tool_name FROM signals) GROUP BY tool_name ORDER BY uses DESC LIMIT 20
