import { CMAModelParams } from './cma_model_params.js';

export interface CMAScenarioGenerationResult {
    qualitative_description: string;
    parameters: Partial<CMAModelParams>;
}

// Keyword-based scenario generation mapping natural language to model parameters.
// To be replaced with LLM-based generation once an API is integrated.
export function generateScenario(query: string): CMAScenarioGenerationResult {
    
    const params: Partial<CMAModelParams> = {
        d: 0.0,
        taup: 1.0,
        taug: 1.0,
        B: 0.05,
        Cm: 0.0,
        toff: 0.0
    };
    
    let description = `A generated scenario based on query: "${query}". `;
    
    // Process query to match the parameter space defined by the model
    const lowerQuery = query.toLowerCase();
    
    if (lowerQuery.includes('sleep in') || lowerQuery.includes('night owl')) {
        params.toff = 2.5;
        description += "This individual is a natural 'night owl' with a delayed sleep phase.";
    } else if (lowerQuery.includes('early bird')) {
        params.toff = -2.0;
        description += "This individual is an 'early bird' with an advanced sleep phase.";
    } else {
        description += "This individual has a standard circadian rhythm and metabolic profile.";
    }
    
    if (lowerQuery.includes('unhealthy') || lowerQuery.includes('diabetes')) {
        params.B = 0.2;
        params.taug = 2.5;
        description += " They have a bias towards higher glucose levels and delayed glucose response.";
    }

    return {
        qualitative_description: description,
        parameters: params
    };
}
