-- Annotation category co-occurrence: pairs of categories assigned to the same trial.
SELECT
    a.category_name as cat_a,
    b.category_name as cat_b,
    count(*) as co_occurrences
FROM annotations a
JOIN annotations b
    ON a.trial_id = b.trial_id
    AND a.category_name < b.category_name
GROUP BY cat_a, cat_b
ORDER BY co_occurrences DESC
