Run the annotation pipeline on extracted signals to classify failure modes.

This runs heuristic annotation first (instant, rule-based), then optionally LLM annotation (reads trajectories), then — if a trained classifier model exists — optionally ensemble or predict annotation.

Input: $ARGUMENTS should be the path to signals file, or defaults to data/signals.jsonl.

Steps:

1. Check the signals file exists. Show trial count.
2. Run heuristic annotation:
   ```
   agent-diagnostics annotate --signals <signals_path> --output data/heuristic.json --annotations-out data/annotations.jsonl
   ```
3. Show annotation summary: how many trials labeled, top categories found.
4. Ask the user if they want LLM annotation too. If yes:
   - Ask how many trials to sample (default 50)
   - Ask which model (haiku is cheapest, sonnet is better, opus is best)
   - Ask which backend: `claude-code` (default, uses the local claude CLI, no API key), `api` (Anthropic SDK, synchronous, needs ANTHROPIC_API_KEY), or `batch` (Message Batches API, 50% cheaper but async, needs ANTHROPIC_API_KEY)
   - Run: `agent-diagnostics llm-annotate --signals <signals_path> --output data/llm.json --sample-size <n> --model <model> --backend <backend> --annotations-out data/annotations.jsonl`
5. Check whether a trained classifier model exists at data/model.json. If it does, ask the user if they want to run classifier annotation. Default to ensemble — predict is only a sanity check.
   - **Ensemble (recommended, production path):** heuristic + classifier two-tier annotation. Ask for the classifier threshold (default 0.5) and the minimum train accuracy to use the classifier (`--min-f1`, default 0.7). Run:
     ```
     agent-diagnostics ensemble --signals <signals_path> --model data/model.json --output data/ensemble.json --threshold <threshold> --min-f1 <min_f1> --annotations-out data/annotations.jsonl
     ```
   - **Predict (sanity check only):** raw classifier predictions, no heuristic tier. Offer this only if the user explicitly wants to inspect the classifier in isolation. Ask for the prediction threshold (default 0.5). Run:
     ```
     agent-diagnostics predict --model data/model.json --signals <signals_path> --output data/predict.json --threshold <threshold> --annotations-out data/annotations.jsonl
     ```
   If no model exists at data/model.json, skip this step and tell the user they can run `/train` first to enable ensemble annotation.
6. Show final summary of all annotations in data/annotations.jsonl:
   ```
   agent-diagnostics query "SELECT annotator_type, count(*) as labels FROM annotations GROUP BY annotator_type"
   ```
