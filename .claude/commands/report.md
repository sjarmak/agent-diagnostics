Generate a reliability report from the annotated dataset.

Steps:

1. Check what annotation files exist (data/heuristic.json, data/llm.json, data/annotations_clean.json, or the file specified in $ARGUMENTS).
2. If multiple exist, use the most complete one (prefer ensemble > llm > heuristic).
3. Run: `agent-diagnostics report --annotations <best_file> --output data/report/`
4. Show key findings from the generated report (read the markdown output).
5. Also run summary queries:
   ```
   agent-diagnostics query "SELECT model, count(*) as trials, round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END)*100, 1) as pass_rate FROM signals GROUP BY model ORDER BY pass_rate DESC"
   ```
   ```
   agent-diagnostics query "SELECT category_name, count(*) as occurrences FROM annotations GROUP BY category_name ORDER BY occurrences DESC LIMIT 15"
   ```
6. Present a concise summary: overall pass rate, top failure categories, model comparison, and where to find the full report.
