Score how well a predictor's emitted confidences match observed outcomes (ECE / Brier / reliability per category).

Use this to check whether the heuristic or LLM annotator's confidence numbers are trustworthy. A predictor is well-calibrated when, of all categories it labels at 0.8 confidence, roughly 80% are actually correct.

Input: $ARGUMENTS may name the predictor annotation file; otherwise default to data/heuristic.json. The reference can be either a ground-truth annotation JSON or a golden corpus directory.

Steps:

1. Check the predictor annotation file exists (default data/heuristic.json). Its categories must carry emitted confidences — heuristic.json and llm.json both do.
2. Determine the reference. Ask the user which they have:
   - A ground-truth annotation JSON file (e.g. data/annotations_clean.json) — pass as `--reference`.
   - A golden corpus directory with per-trial `expected_annotations.json` files — pass as `--golden-dir`.
   Exactly one is required.
3. Run calibration:
   ```
   agent-diagnostics calibrate --predictor <predictor> --reference <reference> --output-dir data/calibration/
   ```
   or, with a golden corpus:
   ```
   agent-diagnostics calibrate --predictor <predictor> --golden-dir <golden_dir> --output-dir data/calibration/
   ```
4. Read data/calibration/calibration.md and present the key results: overall ECE and Brier score, and any categories that are poorly calibrated (large gap between confidence and observed accuracy).
5. If a category is badly miscalibrated, suggest the user inspect its heuristic rules or lower the trust placed in its confidences downstream.
