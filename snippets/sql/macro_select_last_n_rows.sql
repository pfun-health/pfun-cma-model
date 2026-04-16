CREATE OR REPLACE MACRO get_last_rows(num_rows) AS TABLE
SELECT 
    forecasted_events, 
    qualitative_description, 
    parameters, 
    recommendations
FROM cma_recs
LIMIT num_rows 
OFFSET (SELECT count(*) - num_rows FROM cma_recs);


--
-- Usage:
-- -----
select * from get_last_rows(10);
