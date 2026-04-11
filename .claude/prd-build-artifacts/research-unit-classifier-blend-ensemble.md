# Research: unit-classifier-blend-ensemble

## Key Findings

### classifier.py (train function)

- `train()` at line 215 iterates `y_per_cat.items()` at line 261
- Need to skip categories where `derived_from_signal: true` in taxonomy_v3
- Three derived categories: `incomplete_solution`, `near_miss`, `minimal_progress`
- Can use `_package_data_path('taxonomy_v3.yaml')` + `load_taxonomy()` + `_extract_categories()` from taxonomy.py

### ensemble.py (HEURISTIC_ONLY)

- Lines 26-34: frozenset with 5 entries including `near_miss` and `minimal_progress`
- Remove those two since they are `derived_from_signal: true`
- Remaining: `exception_crash`, `rate_limited_run`, `edit_verify_loop_failure`

### blend_labels.py (hardcoded trust list)

- Lines 67-74: hardcoded set `{"rate_limited_run", "exception_crash", "near_miss", "over_exploration", "edit_verify_loop_failure"}`
- Replace with metadata-driven: load taxonomy_v3, find categories where `signal_dependencies` is non-empty
- From taxonomy_v3: only `incomplete_solution`, `near_miss`, `minimal_progress` have non-empty signal_dependencies
- BUT the intent is categories detectable from signals alone = structural/deterministic ones
- Actually re-reading: "find categories where signal_dependencies is non-empty" -- that gives derived categories
- The spec says "use those as the trusted set" for categories that CAN be detected from signals alone
- Wait: the hardcoded list is for "high-confidence structural categories" -- these are heuristic-detectable
- Need to re-read spec: "Replace the hardcoded fallback category list with metadata-driven trust from taxonomy signal_dependencies"
- The taxonomy has `signal_dependencies` field. Categories with `derived_from_signal: false` AND empty `signal_dependencies` are NOT signal-detectable
- Categories with non-empty `signal_dependencies` ARE signal-detectable (derived_from_signal: true ones)
- But the current hardcoded list includes categories like `exception_crash` which have empty signal_dependencies
- Conclusion: The spec says to use signal_dependencies as the trust indicator. Categories with non-empty signal_dependencies list are trusted.

### taxonomy.py

- `_package_data_path(filename)` resolves bundled data files
- `load_taxonomy(path)` loads and caches YAML
- `_extract_categories(taxonomy)` returns flat list of category dicts
- `_is_v3(taxonomy)` checks for v3 format

### Test files

- test_classifier.py: has `_build_synthetic_dataset` helper, need to add derived categories to it
- test_blend_labels.py: has `TestDefaultTrustedCategories` that checks hardcoded defaults -- needs update
- test_ensemble.py: has `TestHeuristicOnly.test_contains_structural_categories` that checks exact set -- needs update
