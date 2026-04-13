# Plan: unit-cooccurrence-dimensions

## Implementation steps

### 1. Add `co_occurrence_matrix(annotations)` to report.py

- Extract set of category names per annotation
- Build pairwise counts: for each pair (A, B), count trials where both appear
- Diagonal = prevalence count for each category
- Compute phi coefficient for each off-diagonal pair
- Return symmetric dict-of-dicts {catA: {catB: phi, catA: count}}

### 2. Add `dimension_aggregation(annotations, taxonomy)` to report.py

- Build category->dimension mapping from taxonomy
- For each dimension, find trials that have at least one category in that dimension
- Compute failure_rate = failed_trials_with_dimension / total_trials_with_dimension
- Return {dimension: {failure_rate, trial_count}}

### 3. Update `_render_markdown()` to include Dimension Summary section

- Add `dimension_summary` parameter
- Render table with dimension name, trial count, failure rate

### 4. Update `generate_report()` to wire everything together

- Call co_occurrence_matrix and dimension_aggregation
- Pass dimension_summary to \_render_markdown
- Add dimension_summary and co_occurrence to JSON output

### 5. Write tests covering all acceptance criteria
