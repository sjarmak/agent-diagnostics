# Plan: unit-report-tests

## Step 1: TestCoreTaskName (parametrized)

- Test cases: bare name, with hash, with each prefix (baseline*, bl*, mcp*, sgonly*), with path dirs, trailing slash, case normalization, double underscore in task name

## Step 2: TestPairedComparison

- Build fixture with 2 configs sharing 25+ tasks, each with categories
- Verify shared_tasks count, introduced_by_a/b structure
- Test < 20 shared tasks returns empty

## Step 3: TestTopCategoriesWithExamples

- Build annotations with known polarity categories
- Verify top_n and examples_per limits
- Verify structure of returned dicts

## Step 4: TestCategoryBySuite

- Build annotations across multiple suites
- Verify grouping, sorting by total desc

## Step 5: TestRenderMarkdownNoSuccesses

- Call \_render_markdown with empty top_successes list
- Verify "No success-mode annotations found" appears in output

## Step 6: Run coverage, fix gaps if needed
