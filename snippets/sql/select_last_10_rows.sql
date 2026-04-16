SELECT 
    forecasted_events, 
    qualitative_description, 
    parameters, 
    recommendations
FROM cma_recs
LIMIT 10 
OFFSET (SELECT count(*) - 10 FROM cma_recs);
