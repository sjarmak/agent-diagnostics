-- Top tool usage from tool_call_sequence arrays.
SELECT
    tool_name,
    count(*) as usage_count
FROM (
    SELECT unnest(tool_call_sequence) as tool_name
    FROM signals
)
GROUP BY tool_name
ORDER BY usage_count DESC
LIMIT 20
