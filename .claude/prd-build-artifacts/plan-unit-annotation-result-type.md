# Plan: unit-annotation-result-type

## Step 1: Add AnnotationResult to types.py

- Add `Union` to typing imports
- Add three frozen dataclasses: AnnotationOk, AnnotationNoCategoriesFound, AnnotationError
- Define AnnotationResult = Union[...] with **doc** set

## Step 2: Add is_error guard to annotation_cache.py

- Add `is_error: bool = False` parameter to `put_cached`
- Early return (skip caching) when `is_error=True`
- Log a warning when skipping

## Step 3: Use AnnotationResult internally in llm_annotator.py

- Import AnnotationResult types from types.py
- Add internal helper `_unwrap_result` that converts AnnotationResult to list[dict]
- Refactor `annotate_trial_claude_code` to create AnnotationError/AnnotationNoCategoriesFound/AnnotationOk internally, then unwrap at boundary
- Refactor `annotate_trial_api` similarly
- Only call put_cached for AnnotationOk results (pass is_error=True otherwise as defense)

## Step 4: Add mypy to pyproject.toml dev deps

## Step 5: Install mypy and run type check

## Step 6: Write tests/test_annotation_result.py

- Test AnnotationOk, AnnotationNoCategoriesFound, AnnotationError construction
- Test AnnotationResult importable with docstring
- Test unwrap behavior for each variant
- Test that put_cached skips on is_error=True

## Step 7: Run all acceptance criteria
