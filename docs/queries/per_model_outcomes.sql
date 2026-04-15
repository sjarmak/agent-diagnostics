-- Per-model outcome summary: pass rate, trial count, and total passed.
SELECT
    model,
    count(*) as trials,
    sum(CASE WHEN passed THEN 1 ELSE 0 END) as passed,
    round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END) * 100, 1) as pass_rate
FROM signals
GROUP BY model
ORDER BY pass_rate DESC
