# Plan: unit-calibrate

## Steps

1. Copy CSB calibrate.py to src/agent_observatory/calibrate.py verbatim
2. Verify no `from observatory.` imports
3. Write tests/test_calibrate.py with:
   - compare_annotations with known TP/FP/FN scenarios
   - cohen_kappa: perfect agreement (k=1), random (~0), perfect disagreement
   - compare_cross_model with known categories
   - format_markdown and format_cross_model_markdown produce valid markdown
   - Edge cases: empty files, no shared trials
4. Run tests, fix failures
5. Commit
