Rebuild the benchmark manifest table from the current signals data.

`manifest refresh` rewrites manifests.jsonl by deriving benchmark/task metadata from data/signals.jsonl. Run it after ingesting new trials so manifest-joined queries and reports stay accurate.

Input: $ARGUMENTS may name an alternate data directory; otherwise default to data/.

Steps:

1. Check that data/signals.jsonl exists (or the signals file under the directory given in $ARGUMENTS). If it is missing, tell the user to run `/ingest` first.
2. Refresh the manifest:
   ```
   agent-diagnostics manifest refresh --data-dir data/
   ```
3. Report the result: how many manifest rows were written and where (data/manifests.jsonl).
4. Confirm the manifests are now consistent with signals — manifest-joined `/explore` queries and `/report` runs will reflect the latest ingested trials.
