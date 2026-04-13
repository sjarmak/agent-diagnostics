# Plan: unit-trajectory-denominators

## Approach

### 1. annotator.py — Add CHECKER_REQUIRES_TRAJECTORY dict

Module-level dict mapping category_name -> bool. True = needs trajectory data.

```python
CHECKER_REQUIRES_TRAJECTORY: dict[str, bool] = {
    "rate_limited_run": False,
    "exception_crash": False,
    "incomplete_solution": False,
    "near_miss": False,
    "minimal_progress": False,
    # All others: True
    ...
}
```

Export this as a public constant.

### 2. report.py — Update \_category_counts and generate_report

- Import CHECKER_REQUIRES_TRAJECTORY from annotator
- Add `_count_trajectory_available(annotations)` helper: count annotations where signals.has_trajectory is True (or via a `has_trajectory` field in the annotation dict)
- Update `_category_counts()` to return `{name: {count: N, denominator: M, rate: N/M}}` instead of `{name: count}`
- Split markdown "Category Frequency" into two subsections:
  - "Trajectory-Dependent Categories" with denominator = trajectory-available count
  - "Reward-Dependent Categories" with denominator = total corpus
- Update JSON to include denominator info

### 3. Backward compatibility

- The annotation dicts passed to report.py need `has_trajectory` or `signals.has_trajectory`
- Check if annotation dicts carry signals; if not, use a heuristic: look for `has_trajectory` key directly in the annotation dict
- For the annotation dict format, annotations may have a `signals` sub-dict with `has_trajectory`

### 4. Tests

- Test denominator splitting with mixed has_trajectory annotations
- Test retrieval_failure uses trajectory denominator
- Test JSON structure includes {count, denominator, rate}
- Test markdown sections exist
