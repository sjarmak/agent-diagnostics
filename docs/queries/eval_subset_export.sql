-- Export failing trials for evaluation subset analysis.
SELECT
    s.trial_id,
    s.task_id,
    s.model,
    s.reward,
    s.passed,
    s.total_turns,
    s.tool_calls_total,
    s.duration_seconds
FROM signals s
WHERE s.passed = false
ORDER BY s.task_id
