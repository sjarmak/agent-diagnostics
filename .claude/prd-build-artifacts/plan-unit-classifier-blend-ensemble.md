# Plan: unit-classifier-blend-ensemble

## Step 1: Modify classifier.py train()

- Import `_package_data_path`, `load_taxonomy`, `_extract_categories` from taxonomy module
- At the start of `train()`, load taxonomy_v3 and extract derived category names
- Before the training loop (line 261), filter `y_per_cat` to exclude derived categories
- Add derived categories to `skipped` list with reason

## Step 2: Modify ensemble.py HEURISTIC_ONLY

- Remove `"near_miss"` and `"minimal_progress"` from the frozenset
- Update docstring/comment to reflect change

## Step 3: Modify blend_labels.py hardcoded trust list

- Import taxonomy loading functions
- Replace hardcoded set (lines 67-74) with dynamic loading from taxonomy_v3
- Load taxonomy, extract categories, find those with non-empty `signal_dependencies`
- Use those category names as the trusted set

## Step 4: Update existing tests

- test_ensemble.py: Update `TestHeuristicOnly.test_contains_structural_categories` to expect 3 items
- test_blend_labels.py: Update `TestDefaultTrustedCategories` and `TestHeuristicOnlyTrusted` and `TestLLMPriority` to use new trusted set

## Step 5: Add new tests

- test_classifier.py: Add `test_derived_categories_excluded` verifying incomplete_solution, near_miss, minimal_progress are excluded
- test_blend_labels.py: Add `test_no_hardcoded_trust` verifying no hardcoded category names in the else branch

## Step 6: Run all tests and fix failures
