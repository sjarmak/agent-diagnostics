# Plan: unit-manifest-integration

## Step 1: Add benchmark_source to TrialSignals (types.py)

- Add `benchmark_source: str` field to the TypedDict

## Step 2: Add load_manifest() function (signals.py)

- Reads MANIFEST.json from a given path
- Parses JSON: expects dict mapping run directory names to benchmark names
- Returns dict[str, str] usable as suite_mapping

## Step 3: Add directory-name convention resolver (signals.py)

- Function: \_resolve_benchmark_from_directory(trial_dir: Path) -> str | None
- Known patterns: "crossrepo" -> "crossrepo", "openhands" -> "openhands", "swe-bench" or "swe_bench" -> "swe-bench"
- Checks directory name parts for known benchmark substrings

## Step 4: Modify \_resolve_benchmark to return source info (signals.py)

- Change return to tuple[str | None, str] where second element is source
- Sources: "manifest" (from suite_mapping), "directory" (from dir convention), "" (not resolved)
- Update caller in extract_signals()

## Step 5: Update extract_signals() (signals.py)

- Use new \_resolve_benchmark return value
- Set benchmark_source in TrialSignals dict

## Step 6: Update test key count expectation (test_signals.py)

- Update expected keys set to include benchmark_source (27 keys now)

## Step 7: Add tests (test_signals.py)

- Test load_manifest: valid JSON, missing file
- Test directory-name fallback: crossrepo, openhands, swe-bench patterns
- Test benchmark_source provenance: manifest, directory, empty string
