# Plan: unit-llm-annotator

## Implementation Steps

1. Create `src/agent_observatory/llm_annotator.py`:
   - Copy CSB source, apply all 7 changes from research
   - Remove judge_trial, \_build_judge_input, \_extract_code_changes (lines 653-832)
   - Remove sys import (only used for sys.path manipulation in judge code; keep for stderr print)
   - Change taxonomy import and \_taxonomy_yaml implementation
   - Rename \_build_prompt -> build_prompt, \_validate_categories -> validate_categories, etc.
   - Update all internal references to use new public names
   - Change anthropic ImportError in annotate_trial_api to raise

2. Create `tests/test_llm_annotator.py`:
   - Test build_prompt output structure
   - Test validate_categories filtering
   - Test truncate_trajectory short/long
   - Test summarise_step compact output
   - Test annotate_trial_llm dispatch
   - Test anthropic ImportError
   - Test judge_trial does NOT exist
   - Test no sys.path in source
   - Test all public imports

3. Run tests, fix failures
