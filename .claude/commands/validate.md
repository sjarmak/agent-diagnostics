Validate an annotation file for schema and integrity errors.

Use this before training, calibrating, or reporting to catch malformed annotations early.

Input: $ARGUMENTS should be the path to the annotation JSON file, or defaults to data/annotations.jsonl.

Steps:

1. Check the annotation file exists (default data/annotations.jsonl). If $ARGUMENTS names a different file, use that.
2. Run validation:
   ```
   agent-diagnostics validate --annotations <annotations_path>
   ```
3. Report the result:
   - If validation passes, tell the user the file is clean and ready for training/reporting.
   - If it fails, show the reported errors verbatim and explain what each one means (e.g. unknown category name, missing primary-key field, taxonomy version mismatch).
4. If errors were found, suggest the fix — usually re-running `/label` against the current taxonomy, or correcting the source signals.
