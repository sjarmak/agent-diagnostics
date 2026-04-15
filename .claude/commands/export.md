Export the dataset to shareable Parquet format.

Steps:

1. Check that data/signals.jsonl exists. Show row count.
2. Run: `agent-diagnostics export --format parquet --out data/export/ --data-dir data/`
3. Show the MANIFEST.json contents (schema version, row counts, file sizes, checksums).
4. Show file sizes of the export: `ls -lh data/export/`
5. Tell the user: "The data/export/ directory is ready to share. It contains Parquet files readable by pandas, Polars, R, DuckDB, or any Arrow-compatible tool."
