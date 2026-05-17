Train per-category classifiers from LLM-labeled data.

This produces a model JSON artifact (not annotations). Train once per LLM-label batch; then use `/label` to run the trained model in ensemble mode for production labeling.

Input: $ARGUMENTS should be the path to the LLM annotation JSON used as training labels, or defaults to data/llm.json (the output of `/label`'s LLM step).

Steps:

1. Check the labels file exists (default data/llm.json). If it is missing, tell the user to run `/label` and accept the LLM annotation step first — training needs LLM labels.
2. Check the signals file exists. Default data/signals.jsonl; ask the user if their signals live elsewhere.
3. Ask the user for training settings, offering the defaults:
   - `--min-positive` — minimum positive examples per category (default 3); categories below this are skipped.
   - `--lr` — learning rate (default 0.1)
   - `--epochs` — training epochs (default 300)
4. Run training with evaluation enabled so the summary is printed:
   ```
   agent-diagnostics train --labels <labels_path> --signals <signals_path> --output data/model.json --min-positive <n> --lr <lr> --epochs <epochs> --eval
   ```
5. Report the training summary from the command output: number of classifiers trained, training sample count, categories skipped (and why — too few positive examples), and per-category train accuracy from the `--eval` block.
6. Tell the user: "Model saved to data/model.json. Run `/label` to apply it in ensemble mode (heuristic + classifier)."
