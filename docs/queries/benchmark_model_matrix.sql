-- Benchmark x model matrix: trials and pass rate per combination.
SELECT
    benchmark,
    model,
    count(*) as trials,
    round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END) * 100, 1) as pass_rate
FROM signals
GROUP BY benchmark, model
ORDER BY benchmark, model
