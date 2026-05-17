Inspect the database schema of the observatory tables.

`db schema` emits the column names and types for the `signals`, `annotations`, and `manifests` tables. Use it as a power-user reference when writing SQL for `/explore` — it shows exactly what columns exist in the current data.

Input: $ARGUMENTS may name an alternate data directory; otherwise default to data/.

Steps:

1. Emit the schema:
   ```
   agent-diagnostics db schema --data-dir data/
   ```
   The default output is markdown. Add `--format json` when the schema needs to be piped into another tool.
2. Present the table/column listing clearly.
3. If the user is about to write a query, point them at `/explore` to run it, and note that the canonical column documentation lives in [docs/schemas/signals.md](../../docs/schemas/signals.md) and [docs/schemas/annotations.md](../../docs/schemas/annotations.md).
