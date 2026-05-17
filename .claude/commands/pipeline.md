Run the stale stages of the analysis pipeline declared in pipeline.toml.

`pipeline run` inspects each stage's declared inputs and outputs and re-runs only the stages whose inputs changed since their outputs were last written — like `make` for the ingest → annotate → report flow.

Input: $ARGUMENTS may name an alternate pipeline config; otherwise default to pipeline.toml at the project root.

Steps:

1. Check that pipeline.toml exists at the project root (or the path given in $ARGUMENTS). If it is missing, tell the user no pipeline is configured and they should run the individual commands (`/ingest`, `/label`, `/report`) directly.
2. Run the pipeline:
   ```
   agent-diagnostics pipeline run
   ```
   If a non-default config is used, add `--config <path>`. If inputs/outputs use relative paths anchored somewhere other than the config's directory, add `--project-root <path>`.
3. Report which stages ran, which were skipped as up-to-date, and any stage failures.
4. If a stage failed, show its error and suggest running that stage's underlying command on its own to debug.
