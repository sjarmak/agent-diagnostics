# Test Results: unit-trajectory-denominators

## Test Run

```
python3 -m pytest tests/test_report.py tests/test_annotator.py -v
166 passed in 0.14s
```

## New Tests Added

### test_annotator.py (7 new tests)

- TestCheckerRequiresTrajectory::test_is_dict
- TestCheckerRequiresTrajectory::test_all_values_are_bool
- TestCheckerRequiresTrajectory::test_retrieval_failure_requires_trajectory
- TestCheckerRequiresTrajectory::test_incomplete_solution_does_not_require_trajectory
- TestCheckerRequiresTrajectory::test_rate_limited_run_does_not_require_trajectory
- TestCheckerRequiresTrajectory::test_clean_success_requires_trajectory
- TestCheckerRequiresTrajectory::test_covers_all_checker_output_categories

### test_report.py (20 new tests)

- TestTrajectoryAvailableCount (5 tests): empty, all, none, mixed, signals sub-dict
- TestDenominatorSplitting (4 tests): traj vs reward denominator, rate computation
- TestRetrievalFailureDenominator (2 tests): metadata flag + report denominator
- TestJsonDenominatorStructure (2 tests): {count, denominator, rate} keys + trajectory_available
- TestMarkdownDenominatorSections (5 tests): section presence, denominator text, Rate column
- TestCheckerRequiresTrajectoryMetadata (3 tests): reward-only false, trajectory true, coverage

### Updated Existing Tests (5 tests)

- TestMarkdownSections::test_category_frequency_section -> split into trajectory + reward tests
- TestCategoryCountsCorrect: updated to check .count instead of bare int
- TestEmptyAnnotations::test_empty_md_has_sections: updated section names

## Acceptance Criteria Coverage

- [x] Each heuristic checker has requires_trajectory classification
- [x] generate_report() shows 'Trajectory-Dependent Categories' section
- [x] generate_report() shows 'Reward-Dependent Categories' section
- [x] retrieval_failure uses trajectory-available trials as denominator
- [x] JSON includes {count, denominator, rate} structure
- [x] Tests cover denominator splitting, retrieval_failure denominator, JSON structure
- [x] All tests pass with 0 failures
