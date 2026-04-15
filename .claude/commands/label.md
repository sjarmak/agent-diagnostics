Run the annotation pipeline on extracted signals to classify failure modes.

This runs heuristic annotation first (instant, rule-based), then optionally LLM annotation (reads trajectories, requires ANTHROPIC_API_KEY).

Input: $ARGUMENTS should be the path to signals file, or defaults to data/signals.jsonl.

Steps:

1. Check the signals file exists. Show trial count.
2. Run heuristic annotation:
   ```
   agent-diagnostics annotate --signals <signals_path> --output data/heuristic.json --annotations-out data/annotations.jsonl
   ```
3. Show annotation summary: how many trials labeled, top categories found.
4. Ask the user if they want LLM annotation too (needs ANTHROPIC_API_KEY). If yes:
   - Ask how many trials to sample (default 50)
   - Ask which model (haiku is cheapest, sonnet is better, opus is best)
   - Ask which backend (batch is 50% cheaper but async, api is synchronous)
   - Run: `agent-diagnostics llm-annotate --signals <signals_path> --output data/llm.json --sample-size <n> --model <model> --backend <backend> --annotations-out data/annotations.jsonl`
5. Show final summary of all annotations in data/annotations.jsonl:
   ```
   agent-diagnostics query "SELECT annotator_type, count(*) as labels FROM annotations GROUP BY annotator_type"
   ```
