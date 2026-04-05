# Test Results: unit-report-tests

## Coverage

- report.py: 99% (219 stmts, 3 missed — lines 23-24, 319)
- Target: >= 85% — PASSED

## Tests Added (32 new tests)

- TestCoreTaskName: 13 parametrized cases covering all prefix patterns
- TestPairedComparison: 7 tests including shared tasks, deltas, <20 threshold, top5 limit
- TestTopCategoriesWithExamples: 5 tests for polarity filtering, top_n, examples_per
- TestCategoryBySuite: 4 tests for grouping, sorting, unknown benchmark, empty
- TestRenderMarkdownNoSuccesses: 3 tests for no-success branch, with-success, paired comparisons

## Full Suite

- 377 tests passed, 0 failures
